"""Switch-off ablations of submission #337501 (Table 4 of the technical write-up).

Usage:
    python ablation/ablate.py <variant> <net0> <net1> <out.jsonl> [--dataset DIR]

Runs the exact submitted estimator (submission-337501/) in-process under flopscope, with one component switched off, on
networks net0..net1-1 of the public `full` split (all-layer means from 1e9 samples), and appends one JSON line per
network to out.jsonl: the final-layer MSE, the label-noise term avg_variance / 1e9, the noise-corrected MSE and the
FLOPs billed by flopscope. A component is switched off by overriding its parameter defaults in PARAMETER_REGISTRY before
`setup` (setup builds the parameters from the registry), or, for three variants, by an exact source patch. Nothing is
retrained. `base` runs the unmodified estimator. The write-up used networks 896-959; `ablation/summarize.py` computes
the paired ratios. A run resumes from an existing out.jsonl.

By default the `full` split is loaded with whestbench from hf://aicrowd/arc-whestbench-public-2026 (revision
v2-phase2), which downloads the prepared split once (about 63 GB). --dataset DIR loads a local copy of that prepared
split (the folder saved with `datasets.Dataset.save_to_disk`) instead.
"""
import argparse, os, sys, json, time, importlib.util
sys.dont_write_bytecode = True   # keep submission-337501/ byte-identical (no __pycache__)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.domain import MLP
from whestbench.sdk import SetupContext

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'submission-337501')
FOLDER = os.path.normpath(FOLDER)
N_SAMPLES = 10 ** 9   # Monte Carlo samples behind the labels of the `full` split

VARIANTS = {   # registry default overrides; the registry documents 0 as switching the component off
    'base': {},
    'r8_off': {'r8_output_scale': [0.0, 0.0, 0.0]},
    'v2e_off': {'k21_slot_v2e_lambda': 0.0},
    'joint_off': {'joint_amplitude': 0.0},
    'feed_off': {'feed_fold': 0},
    'h12_off': {'h12_beta_layers': 'ZEROS', 'h12_gamma_layers': 'ZEROS'},
    'term_k3_off': {'final_k3_scale': 0.0},
    'term_k4_off': {'final_k4_scale': 0.0},
    'k4_star_off': {'k4_star_scale': 0.0},
    'v2e_slot_off': {'k21_slot': 0},
    'h12_fold_off': {'h12_fold_gain': 0.0},
    'k31_fold_off': {'exact_k31_betas': 'ZEROS'},
    'k4_vertices_off': {'k4_d22c_scale': 0.0, 'k4_d31c_scale': 0.0},
    'dress_off': {'k3_vertex_dress': 0.0, 'k4_vertex_dress': 0.0},
    'pair_terms_off': {'cov_k3cross': 0.0, 'cov_k3sq': 0.0, 'split_pair_c3_cov': 0.0},
    'edgeworth34_off': {'edgeworth_34': 0.0},
    'k4_host_cov_off': {'k4_cov_weight': 0.0},
    'sphere_off': {},
    'geometry_off': {},
    'all_learned_off': {'r8_output_scale': [0.0, 0.0, 0.0], 'k21_slot_v2e_lambda': 0.0, 'joint_amplitude': 0.0},
    'joint_geometry_off': {'joint_amplitude': 0.0},
    'births_all_off': {'h12_fold_gain': 0.0, 'feed_fold': 0, 'h12_beta_layers': 'ZEROS', 'h12_gamma_layers': 'ZEROS', 'fe30': {'coef': [0.0, 0.0]}, 'exact_k31_betas': 'ZEROS'},
}
SRC_PATCH = {   # exact source replacements: Gaussian input instead of the sphere; Geometry output scale set to 0
    'sphere_off': [('return fnp.asarray(0.9997559189796448, dtype=fnp.float32)', 'return fnp.asarray(1.0, dtype=fnp.float32)'),
                   ('k4_core = -6.0 / float(width + 2)', 'k4_core = 0.0 / float(width + 2)'),
                   ("runtime.coefficients['input_core'] = _ops5_pub_asarray(-6.0 / float(width + 2), dtype=fnp.float32)",
                    "runtime.coefficients['input_core'] = _ops5_pub_asarray(0.0 / float(width + 2), dtype=fnp.float32)")],
    'geometry_off': [('dscale = _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]',
                      'dscale = 0.0 * _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]')],
    'all_learned_off': [('dscale = _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]',
                      'dscale = 0.0 * _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]')],
    'joint_geometry_off': [('dscale = _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]',
                      'dscale = 0.0 * _ops5_pub_concatenate([4.0 / rw, _ops5_pub_asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]')],
}
ap = argparse.ArgumentParser(description='Switch-off ablations of submission #337501.')
ap.add_argument('variant', choices=sorted(VARIANTS))
ap.add_argument('net0', type=int)
ap.add_argument('net1', type=int)
ap.add_argument('out')
ap.add_argument('--dataset', default=None, help='local folder of the prepared `full` split (default: download from the Hub)')
args = ap.parse_args()
VARIANT, N0, N1, OUT = args.variant, args.net0, args.net1, args.out

if args.dataset:
    from datasets import Dataset
    ds = Dataset.load_from_disk(args.dataset)
else:
    import whestbench
    ds = whestbench.load_dataset('aicrowd/arc-whestbench-public-2026', revision='v2-phase2', split='full')
    assert int(whestbench.metadata(ds)['n_samples']) == N_SAMPLES

spec = importlib.util.spec_from_file_location('est_' + VARIANT, os.path.join(FOLDER, 'estimator.py'))
m = importlib.util.module_from_spec(spec)
if VARIANT in SRC_PATCH:
    src = open(os.path.join(FOLDER, 'estimator.py')).read()
    for old, new in SRC_PATCH[VARIANT]:
        assert src.count(old) >= 1, old
        src = src.replace(old, new)
    m.__file__ = os.path.join(FOLDER, 'estimator.py'); sys.modules[spec.name] = m
    exec(compile(src, m.__file__, 'exec'), m.__dict__)
else:
    spec.loader.exec_module(m)
for k, v in VARIANTS[VARIANT].items():
    entry = m.PARAMETER_REGISTRY[k]
    if v == 'ZEROS':
        d = entry['default']; v = {kk: 0.0 for kk in d} if isinstance(d, dict) else type(d)(0.0 for _ in d)
    entry['default'] = v
    if 'allowed' in entry and entry['allowed'] is not None and v not in entry['allowed']:
        entry['allowed'] = list(entry['allowed']) + [v]
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done.add(json.loads(line)['net'])
        except Exception: pass
E = m.Estimator()
with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
    E.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, seed=0, api_version='v2-phase2', submission_dir=FOLDER))
check = {k: (m.PARAMS.get(k) if not isinstance(m.PARAMS.get(k), (list, tuple)) else list(m.PARAMS.get(k))[:3]) for k in VARIANTS[VARIANT]}
for net in range(N0, N1):
    if net in done:
        continue
    row = ds[net]
    raw = np.asarray(row['weights'], dtype=np.float32)
    target = np.asarray(row['all_layer_means'], dtype=np.float64)
    assert raw.shape == (16, 1024, 1024) and target.shape == (16, 1024)
    t0 = time.perf_counter(); rec = dict(variant=VARIANT, net=net, params_after_setup=check)
    try:
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
            mlp = MLP(width=1024, depth=16, seed=int(row['mlp_seed']), weights=[fnp.array(x) for x in raw])
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as b:
            pred = E.predict(mlp, 2 ** 41)
            vals = np.asarray(pred.tolist() if hasattr(pred, 'tolist') else pred, dtype=np.float64)
        noise = float(row['avg_variance']) / N_SAMPLES
        mse = float(np.mean((vals[-1] - target[-1]) ** 2))
        rec.update(mse=mse, noise=noise, mse_nc=mse - noise, flops=int(b.flops_used), finite=bool(np.isfinite(vals).all()))
    except Exception as e:
        rec.update(error=type(e).__name__ + ': ' + str(e)[:300])
    rec['wall_s'] = round(time.perf_counter() - t0, 1)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
