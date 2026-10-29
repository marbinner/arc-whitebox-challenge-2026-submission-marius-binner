"""Learned residual heads on carried cumulants and geometry.
Inputs: corrector/FE31/V2e/joint assets and public runtime features. Outputs:
corrected cumulants, mean/covariance deltas and carried head state. R8 uses
standardized marginals and pair inputs; V2e adds rank-eight slices. Knobs:
registered head scales/gates, feature selection and swappable implementations."""
from __future__ import annotations
from .array_ops import pad_axis, zero_diagonal_owned
from .config import PARAMS
from .registry import ASSET_REGISTRY
from .state import CORRECTOR_KEYS, HEAD_CONSTANT, W11, W21, flops, fnp, math




class RegisteredWeights(flops.Module):
    def __init__(self, group):
        prefix = group + '.'
        for name, spec in ASSET_REGISTRY.items():
            if name.startswith(prefix):
                dtype = fnp.float64 if spec['dtype'] == 'float64' else fnp.float32
                setattr(self, name[len(prefix):], fnp.zeros((), dtype=dtype))

def load_corrector(runtime, directory):
    runtime.corrector_layers = None
    if directory is None:
        return
    from pathlib import Path as _P
    path = _P(str(directory)) / 'corrector.npz'
    if not path.exists():
        return
    w = RegisteredWeights.from_file(str(path), group='corrector')
    stacked = {k: getattr(w, k) for k in CORRECTOR_KEYS}
    runtime.corrector_layers = [{k: v[layer] for (k, v) in stacked.items()} for layer in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    runtime.learned['corr_stacked'] = stacked

def load_v2e(runtime, directory, width, depth):
    runtime.v2e_model = None
    runtime.v2e_output = None
    if not int(PARAMS['k21_slot']) or int(width) != 1024 or int(depth) != 16:
        return
    from pathlib import Path as _V2ePath
    if directory is None:
        raise ValueError('V2e requires submission_dir and k21_v2e.npz')
    (rank, hidden) = (8, 128)
    w = RegisteredWeights.from_file(str(_V2ePath(str(directory)) / 'k21_v2e.npz'), group='k21_v2e')
    expected = {'W0': (17, hidden), 'b0': (16, hidden), 'W1': (hidden, hidden), 'b1': (hidden,), 'mean': (16, 17), 'std': (16, 17)}
    for (key, shape) in expected.items():
        value = getattr(w, key)
        if tuple(value.shape) != shape or str(value.dtype) != 'float32':
            raise ValueError('V2e weight shape/dtype mismatch: ' + key)
    out_width = int(w.bout.shape[0]) if len(w.bout.shape) == 1 else -1
    if out_width != 6 * rank or tuple(w.Wout.shape) != (hidden, out_width) or str(w.Wout.dtype) != 'float32' or (str(w.bout.dtype) != 'float32'):
        raise ValueError('V2e output weights must be full or packed for the literal term set')
    (Wo, bo) = (w.Wout, w.bout)
    runtime.v2e_output = (Wo, bo)
    runtime.v2e_model = w

def setup_learned_heads(runtime, width, depth):
    if int(width) != 1024 or int(depth) != 16:
        return
    runtime.learned['corr'] = derive_corrector(runtime)
    runtime.learned['corr_in'] = {}
    prepare_corrector_inputs(runtime, int(depth))
    for layer in range(int(depth)):
        corrector_input_weights(runtime, layer, (layer + 1 < int(depth), layer > 0,
            runtime.feature_layers is not None and layer > 0, False))
    if runtime.v2e_model is not None:
        gates = tuple((float(x) * float(PARAMS['k21_slot_v2e_lambda']) for x in PARAMS['k21_slot_v2e_gates']))
        runtime.learned['v2e'] = derive_v2e(runtime, int(width), gates)
    runtime.learned['joint'] = derive_joint(runtime, int(width), int(depth))

def centre_weights(M):
    M64 = M.astype(fnp.float64)
    return (M64 - fnp.mean(M64, axis=1, keepdims=True)).astype(fnp.float32)

def derive_corrector(runtime):
    st = runtime.learned['corr_stacked']
    nl = int(st['bin'].shape[0])
    K = HEAD_CONSTANT
    out = {}
    for b in [0, 1]:
        out['Wc%d' % b] = fnp.unstack(centre_weights(st['Wc%d' % b]), axis=0)
        Wf = centre_weights(st['Wf%d' % b])
        out['Wf%d' % b] = fnp.unstack(pad_axis(Wf, 0, 1, 2), axis=0)
        out['bf%d' % b] = fnp.unstack(fnp.concatenate([st['bf%d' % b], fnp.ones((nl, 1), dtype=fnp.float32) * K], axis=1), axis=0)
        out['Wg%d' % b] = fnp.unstack(fnp.concatenate([st['Wg%d' % b], (st['bg%d' % b] * (0.5 / K))[:, None, :]], axis=1), axis=0)
    out['Wom'] = fnp.unstack(centre_weights(st['Wom']), axis=0)
    tw = int(st['Wom'].shape[2])
    abc = st['ABC']
    nm = int(abc.shape[2]) // 3
    if int(abc.shape[1]) == tw - 3:
        pad = fnp.zeros((nl, 3, nm), dtype=fnp.float32)
        out['ABC3'] = fnp.unstack(fnp.stack([fnp.concatenate([pad, abc[:, :, i * nm:(i + 1) * nm]], axis=1) for i in [0, 1, 2]], axis=1), axis=0)
    sel = [[PARAMS['r8_output_scale'][0], 0.0, 0.0], [0.0, PARAMS['r8_output_scale'][1], 0.0], [-PARAMS['r8_output_scale'][2], 0.0, PARAMS['r8_output_scale'][2]]] + [[0.0, 0.0, 0.0]] * (tw - 3)
    msel = fnp.asarray(sel, dtype=fnp.float32)
    return [dict({k: v[l] for (k, v) in out.items()}, Msel=msel) for l in range(nl)]


def prepare_corrector_inputs(runtime, depth):
    st = runtime.learned['corr_stacked']
    wnew = runtime.learned.get('fe31_wnew')
    fe31 = runtime.feature_layers is not None and wnew is not None
    base = st['Wxrm']
    parts = [base[:, :19]]
    if 'Wxf' in st:
        parts.append(st['Wxf'][:, [0, 1, 5]])
    parts.append(st['bin'][:, None, :])
    parts.append(base[:, 19:])
    if fe31:
        parts.append(wnew)
    rows = fnp.unstack(fnp.concatenate(parts, axis=1), axis=0)
    cache = runtime.learned.setdefault('corr_in', {})
    for layer in range(1, depth - 1):
        cache[layer, (True, True, fe31, False)] = rows[layer]

def corrector_input_weights(runtime, layer, combo):
    cache = runtime.learned.setdefault('corr_in', {})
    found = cache.get((layer, combo))
    if found is not None:
        return found
    (has_c, carry, fe31, has_y) = combo
    w = runtime.corrector_layers[layer]
    base = (w['Wxrm'] if has_c else w['Wxm']) if carry else w['Wxr'] if has_c else w['Wx']
    ncols = 19 if has_c else 9
    parts = [base[:ncols]]
    if 'Wxf' in w:
        idx = [0, 1] + ([2, 3, 4] if has_y else []) + [5]
        parts.append(w['Wxf'][idx])
    parts.append(w['bin'][None, :])
    if carry:
        parts.append(base[ncols:])
    if fe31:
        parts.append(runtime.feature_layers[layer][0])
    out = fnp.concatenate(parts, axis=0)
    cache[layer, combo] = out
    return out

def centred_weight_squared(runtime, W2, n):
    found = runtime.readout.get('w2c')
    if found is not None and found[0] is W2:
        return found[1]
    value = W2 - 2.0 / float(n)
    runtime.readout['w2c'] = (W2, value)
    return value

def standardized_birth_inputs(runtime, layer, rows, variance):
    legs = runtime.features.get('legs')
    hrow = runtime.features.get('hrow')
    dx = runtime.features.get('dx') if runtime.features.get('dx_layer') == layer else None
    h30 = runtime.features.get('hrow30') if runtime.features.get('hrow30_layer') == layer else None
    parts = [fnp.zeros((rows, 4), dtype=fnp.float32) if hrow is None else hrow, fnp.zeros((rows, 2), dtype=fnp.float32) if legs is None else legs]
    if dx is None:
        parts.append(fnp.zeros((rows, 1), dtype=fnp.float32))
    else:
        vv = fnp.maximum(variance.astype(fnp.float64), 1e-10)
        parts.append((dx / (vv * vv))[:, None])
    parts.append(fnp.zeros((rows, 4), dtype=fnp.float32) if h30 is None else h30)
    raw = fnp.concatenate(parts, axis=1)
    if str(raw.dtype) != 'float64':
        raw = raw.astype(fnp.float64)
    scale = fnp.maximum(fnp.mean(fnp.abs(raw), axis=0, keepdims=True), 1e-30)
    raw = raw / scale
    (_, mu, sd) = runtime.feature_layers[layer]
    return ((raw - mu) / sd).astype(fnp.float32)

def corrector_feature_context(runtime, layer, depth, mu, k3, k4, C, P, K, C_prev, W, carry, xf, gaussian, weight_squared):
    from .features import FeatureContext
    return FeatureContext(runtime=runtime, layer=layer, depth=depth, mu=mu, k3=k3, k4=k4,
                          C=C, P=P, K=K, C_prev=C_prev, W=W, carry=carry, xf=xf,
                          gaussian=gaussian, weight_squared=weight_squared)


def correct_layer(runtime, layer, depth, mu, k3, k4, C, P, K, C_prev, W, carry, xf, gaussian, weight_squared):
    from .features import build_corrector_features
    if runtime.corrector_config['implementation'] == 'none':
        return None, None
    ctx = corrector_feature_context(runtime, layer, depth, mu, k3, k4, C, P, K, C_prev, W, carry, xf, gaussian, weight_squared)
    features = build_corrector_features(ctx, runtime.corrector_config['features'])
    return CORRECTOR_IMPLEMENTATIONS[runtime.corrector_config['implementation']](ctx, features)


def r8_corrections(ctx, features):
    (runtime, layer, depth, mu, k3, k4, C, P, K, C_prev, W, carry, xf, gaussian, weight_squared) = (ctx.runtime, ctx.layer, ctx.depth, ctx.mu, ctx.k3, ctx.k4, ctx.C, ctx.P, ctx.K, ctx.C_prev, ctx.W, ctx.carry, ctx.xf, ctx.gaussian, ctx.weight_squared)
    w = runtime.corrector_layers[layer]
    dw = runtime.learned['corr'][layer]
    ones = runtime.readout['ones']
    (v, s, a, Phi, a2, phi, inv_s, inv_v, v2, vs) = gaussian
    parts = features['parts']
    F = parts[0]
    mem = features['matrices'].get('carry')
    fe31 = features['matrices'].get('birth_inputs') is not None
    has_y = False
    if tuple(ctx.runtime.corrector_config['features']) == DEFAULT_CORRECTOR_FEATURES:
        Win = corrector_input_weights(runtime, layer, (C is not None, carry is not None, fe31, has_y))
    else:
        Win = configured_input_weights(ctx, features)
    e = (fnp.concatenate(parts, axis=1) if len(parts) > 1 else F) @ Win
    scale_n = 1.0 / float(e.shape[0])
    for b in [0, 1]:
        sd = fnp.std(e, axis=1)
        e = e + fnp.einsum('i,ij,jk->k', scale_n / sd, e, dw['Wc%d' % b])
        sd = fnp.std(e, axis=1, keepdims=True)
        g = e @ dw['Wf%d' % b] / sd + dw['bf%d' % b]
        u = fnp.polyval([fnp.einsum('ij,ij,ij->ij', g, g, g), g], runtime.coefficients['r8_cubic'])
        g = g * (fnp.tanh(u) + 1.0)
        e = e + g @ dw['Wg%d' % b]
    sd = fnp.std(e, axis=1, keepdims=True)
    T = fnp.tanh(e @ dw['Wom'] / sd + w['bom'])
    carry = (layer, Phi * 0.5, T, mem)
    if layer == 0:
        return (None, carry)
    dm = fnp.einsum('im,im,i,i->i', fnp.stack([ones, a, a2], axis=1), T @ dw['Msel'], phi, s)
    return (dm, carry)


def configured_input_weights(ctx, features):
    from .features import FEATURE_REGISTRY
    runtime, layer = ctx.runtime, ctx.layer
    w = runtime.corrector_layers[layer]
    base = (w['Wxrm'] if ctx.C is not None else w['Wxm']) if ctx.carry is not None else w['Wxr'] if ctx.C is not None else w['Wx']
    ncols = 19 if ctx.C is not None else 9
    supplied = runtime.feature_weights
    parts = []
    offset = 0
    for name in features['names']:
        if name == 'marginals': value = base[:9]
        elif name == 'pairs': value = base[9:19] if ctx.C is not None else None
        elif name == 'physics': value = w['Wxf'][[0, 1, 5]] if 'Wxf' in w else None
        elif name == 'bias': value = w['bin'][None, :]
        elif name == 'carry': value = base[ncols:] if ctx.carry is not None else None
        elif name == 'birth_inputs': value = runtime.feature_layers[layer][0] if layer != 0 and runtime.feature_layers is not None else None
        else:
            width = FEATURE_REGISTRY[name]['width']
            columns = [column - offset for column in features['columns']
                       if offset <= column < offset + width]
            value = supplied[name][layer][columns] if columns else None
        if value is not None: parts.append(value)
        offset += FEATURE_REGISTRY[name]['width']
    if not parts:
        return fnp.zeros((0, int(w['bin'].shape[0])), dtype=fnp.float32)
    return fnp.concatenate(parts, axis=0)


def linear_corrections(ctx, features):
    runtime, layer, depth, mu, k3, k4, C, P, K, C_prev, W, carry, xf, gaussian, weight_squared = (
        ctx.runtime, ctx.layer, ctx.depth, ctx.mu, ctx.k3, ctx.k4, ctx.C, ctx.P, ctx.K,
        ctx.C_prev, ctx.W, ctx.carry, ctx.xf, ctx.gaussian, ctx.weight_squared)
    w = runtime.corrector_layers[layer]
    dw = runtime.learned['corr'][layer]
    ones = runtime.readout['ones']
    (v, s, a, Phi, a2, phi, inv_s, inv_v, v2, vs) = gaussian
    parts = features['parts']
    X = fnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    weights = runtime.linear_model['weight'][layer][features['columns']]
    T = X @ weights + runtime.linear_model['bias'][layer]
    mem = features['matrices'].get('carry')
    carry = (layer, Phi * 0.5, T, mem)
    if layer == 0:
        return None, carry
    dm = fnp.einsum('im,im,i,i->i', fnp.stack([ones, a, a2], axis=1), T @ dw['Msel'], phi, s)
    return dm, carry




DEFAULT_CORRECTOR_FEATURES = ('marginals', 'pairs', 'physics', 'bias', 'carry', 'birth_inputs')
CORRECTOR_CONFIG = {'implementation': 'r8', 'features': ('marginals', 'pairs', 'physics', 'bias', 'carry', 'birth_inputs')}
CORRECTOR_IMPLEMENTATIONS = {'r8': r8_corrections, 'none': None, 'linear': linear_corrections}


def configure_corrector(runtime, config=None):
    from .features import FEATURE_REGISTRY, feature_width, validate_features
    choices = dict(CORRECTOR_CONFIG)
    if config:
        if set(config) - set(choices): raise ValueError('Unknown corrector configuration key')
        choices.update(config)
    if choices['implementation'] not in CORRECTOR_IMPLEMENTATIONS:
        raise ValueError('Unknown corrector implementation')
    choices['features'] = validate_features(choices['features'])
    runtime.corrector_config = choices
    if not hasattr(runtime, 'feature_weights'): runtime.feature_weights = {}
    for name in choices['features']:
        if name not in DEFAULT_CORRECTOR_FEATURES and name not in runtime.feature_weights:
            runtime.feature_weights[name] = fnp.zeros((16, FEATURE_REGISTRY[name]['width'], 64), dtype=fnp.float32)
    if choices['implementation'] == 'linear' and getattr(runtime, 'linear_model', None) is None:
        runtime.linear_model = dict(weight=fnp.zeros((16, feature_width(choices['features']), 19), dtype=fnp.float32),
                                    bias=fnp.zeros((16, 19), dtype=fnp.float32))
    if getattr(runtime, 'linear_model', None) is not None:
        if tuple(runtime.linear_model['weight'].shape) != (16, feature_width(choices['features']), 19):
            raise ValueError('Linear asset does not match configured features')
    return choices



def derive_v2e(runtime, width, gl):
    w = runtime.v2e_model
    (Wo, bo) = runtime.v2e_output
    terms = ['lr', 'P', 'rho']
    hidden = int(w.W1.shape[0])
    nl = int(w.mean.shape[0])
    rw = math.sqrt(float(width))
    K = HEAD_CONSTANT
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15]
    scale = [0.3333333333333333, 1.0, 2.5 / math.sqrt(2.0 * math.pi), 1.0, 10.0, 10.0, float(width), float(width), rw, rw, 1.0, 1.0 / rw]
    const = [[0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [l / 15.0] for l in range(nl)]
    W0 = w.W0.astype(fnp.float64)
    mean = w.mean.astype(fnp.float64)
    std = w.std.astype(fnp.float64)
    S = fnp.asarray(scale, dtype=fnp.float64) / std[:, cols]
    Wm = S[:, :, None] * W0[cols][None, :, :]
    bias = w.b0.astype(fnp.float64) + (fnp.asarray(const, dtype=fnp.float64) - mean) / std @ W0
    unit = fnp.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]] + [[K]]] * nl, dtype=fnp.float64)
    W0f = fnp.concatenate([fnp.concatenate([Wm, bias[:, None, :]], axis=1), unit], axis=2)
    W0l = fnp.unstack(W0f.astype(fnp.float32), axis=0)
    W1 = w.W1.astype(fnp.float64)
    b1 = w.b1.astype(fnp.float64)
    W1a = fnp.concatenate([pad_axis(W1, 0, 1, 1), pad_axis(b1 / K, 0, 1, 0)[None, :]], axis=0).astype(fnp.float32)
    Wfull = fnp.concatenate([Wo.astype(fnp.float64), (bo.astype(fnp.float64) / K)[None, :]], axis=0)
    WA = fnp.stack([Wfull[:, t * 2 * 8:t * 2 * 8 + 8] for t in [0, 1, 2]], axis=0)
    WBT = fnp.stack([Wfull[:, t * 2 * 8 + 8:(t + 1) * 2 * 8].T for t in [0, 1, 2]], axis=0)
    g = fnp.asarray([[[[x / math.sqrt(8)]]] for x in gl[:nl]], dtype=fnp.float64)
    WAl = fnp.unstack((WA[None] * g).astype(fnp.float32), axis=0)
    out = {'W1': W1a, 'W0': {}, 'WA': {}, 'WBT': WBT.astype(fnp.float32), 'rank': 8, 'nt': 3, 'terms': terms}
    for layer in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        out['W0'][int(layer)] = W0l[int(layer)]
        out['WA'][int(layer)] = WAl[int(layer)]
    return out

def apply_v2e(runtime, layer, mean_pre, variance, k3, k4, P, covariance, gaussian=None):
    V = runtime.learned['v2e']
    width = int(P.shape[0])
    v = variance
    if gaussian is None:
        s = fnp.sqrt(v)
        alpha = mean_pre / s
        Phi = flops.stats.norm.cdf(alpha).astype(fnp.float32)
    else:
        (s, alpha, Phi) = gaussian[1:4]
    vs = v * s if gaussian is None else gaussian[9]
    v2 = v * v if gaussian is None else gaussian[8]
    centered_Phi = 2.0 * Phi - 1.0
    ephi = fnp.exp(-0.5 * alpha * alpha)
    unit = fnp.einsum('i,j->ij', v, s, optimize=False)
    Pbn = zero_diagonal_owned(P / unit)
    rho = zero_diagonal_owned(covariance / fnp.einsum('i,j->ij', s, s, optimize=False))
    rr = fnp.einsum('ij,ij->i', rho, rho)
    X = fnp.stack([alpha, centered_Phi, ephi, fnp.log(v / fnp.mean(v)), k3 / vs, k4 / v2, fnp.einsum('ij,ij->i', Pbn, Pbn), fnp.einsum('ij,ij->j', Pbn, Pbn), fnp.sum(Pbn, axis=1), fnp.sum(Pbn, axis=0), rr, rho @ centered_Phi, runtime.readout['ones']], axis=1)
    h = X @ V['W0'][layer]
    h = 0.5 * h * (1.0 + fnp.tanh(h * fnp.polyval(runtime.coefficients['v2e_gelu'], h * h)))
    z = h @ V['W1']
    h = 0.5 * z * (1.0 + fnp.tanh(z * fnp.polyval(runtime.coefficients['v2e_gelu'], z * z))) + h
    products = fnp.einsum('ik,tkr->tir', h, V['WA'][layer]) @ fnp.einsum('trk,ik->tri', V['WBT'], h)
    mask = fnp.not_equal(P, 0.0)
    mu = unit * mask
    mods = []
    for term in V['terms']:
        if term == 'lr':
            mods.append(mu)
        elif term == 'P':
            mods.append(P)
        else:
            mods.append(rho * mu)
    M = mods[0][None] if len(mods) == 1 else fnp.stack(mods, axis=0)
    delta = fnp.einsum('tij,tij->ij', products, M)
    return P + zero_diagonal_owned(delta)

def derive_joint(runtime, width, depth):
    JW = {k: fnp.asarray(v, dtype=fnp.float64) for (k, v) in runtime.joint_weights.items()}
    Bw = JW['birth.weight']
    H = int(Bw.shape[0])
    nl = int(JW['layer_bias'].shape[0])
    BT = fnp.unstack(Bw.T, axis=0)
    n = float(width)
    rn = math.sqrt(n)
    zero = fnp.zeros((H,), dtype=fnp.float64)
    K = HEAD_CONSTANT
    base = fnp.stack([BT[0] / 3.0, BT[1], 2.5 * BT[2] - BT[6], BT[3], rn * BT[4], BT[6], rn / 3.0 * BT[7], n / 8.0 * BT[8], n / 24.0 * BT[9], n / 4.0 * BT[10], zero, zero], axis=0)
    lcol = fnp.asarray([[l / 15.0] for l in range(nl)], dtype=fnp.float64)
    bias = JW['birth.bias'] - 0.5 * BT[1] + JW['layer_bias'] + lcol * BT[5][None, :]
    Wl = fnp.concatenate([base[None, :, :] * fnp.ones((nl, 1, 1), dtype=fnp.float64), bias[:, None, :]], axis=1)
    kcol = fnp.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]] + [[K]]] * nl, dtype=fnp.float64)
    Wl = fnp.unstack(fnp.concatenate([Wl, kcol], axis=2).astype(fnp.float32), axis=0)
    Ws = fnp.stack([zero, BT[11], 2.5 * BT[12], zero, rn * BT[14], zero, zero, n / 8.0 * BT[15], zero, n / 4.0 * BT[16], 0.5 * BT[13], BT[17], -0.5 * BT[11]], axis=0)
    Ws = pad_axis(Ws, 0, 1, 1).astype(fnp.float32)
    cols = [0, 1, 2, 3, 4, 4, 5, 6, 7, 7]
    Wh = fnp.concatenate([JW['head.weight'].T[:, cols], JW['head.bias'][cols][None, :]], axis=0).astype(fnp.float32)
    hs = (fnp.asarray(runtime.joint_weights['head_scale'], dtype=fnp.float32) * float(PARAMS['joint_amplitude']))[cols]
    ha = fnp.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=fnp.float32)
    G = JW['geometry_weight']
    order = list(range(H)) + [H + 72] + list(range(H, H + 72))
    rscale = [[0.25]] * (H + 1) + [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]] + [[0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333]] + [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    Wg = G.transpose(0, 2, 1)[:, :, [0, 1, 2, 3, 3]][:, order, :] * fnp.asarray(rscale, dtype=fnp.float64)
    # Geometry has no carried feature state: each of its three carried blocks
    # was identically zero. Keep the folded hidden/bias and live extra rows.
    geometry_rows = list(range(H + 1)) + list(range(H + 1, H + 13)) + list(range(H + 25, H + 37)) + list(range(H + 49, H + 61))
    Wg = Wg[:, geometry_rows, :]
    Wg = fnp.unstack(Wg.astype(fnp.float32), axis=0)
    rw = fnp.sqrt(fnp.asarray([float(width)], dtype=fnp.float32))
    milli4 = fnp.asarray([0.001], dtype=fnp.float32) * 4.0
    dscale = fnp.concatenate([4.0 / rw, fnp.asarray([4.0 / float(width)], dtype=fnp.float32), milli4, milli4, milli4])[None, :]
    dadd = fnp.asarray([[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=fnp.float32)
    return {'Wh_terminal': Wh[:, :8], 'hs_terminal': hs[:8], 'ha_terminal': ha[:8], 'Wl': Wl, 'Ws': Ws, 'Wh': Wh, 'hs': hs, 'ha': ha, 'Wg': Wg, 'dscale': dscale, 'dadd': dadd, 'layer_bias': JW['layer_bias'], 'geometry_weight': G}

def read_joint(runtime, k3, k4, layer, wick, ones, gaussian, terminal=False):
    J = runtime.learned['joint']
    (v, sigma, alpha_raw, _, _, _, _, _, v2, vs) = gaussian
    _pb_2 = wick[W11]
    _pb_3 = wick[W21]
    _pb_7 = fnp.mean
    _pb_12 = fnp.stack
    _pb_8 = _pb_7(v)
    _pb_4 = _pb_3 * sigma
    _pb_5 = k3 / vs
    (_pb_11, _pb_14, _pb_16) = fnp.unstack(fnp.multiply([alpha_raw, _pb_5, _pb_5], [alpha_raw, alpha_raw, _pb_5], dtype=fnp.float32), axis=0)
    _pb_6 = k4 / v2
    _pb_9 = v / _pb_8
    _pb_10 = _pb_9 - 1.0
    (_pb_13, _pb_15, _pb_17) = fnp.unstack(fnp.multiply([_pb_11, _pb_6, _pb_10], [_pb_4, alpha_raw, _pb_10], dtype=fnp.float32), axis=0)
    _pb_18 = _pb_12([alpha_raw, _pb_2, _pb_4, _pb_10, _pb_5, _pb_13, _pb_14, _pb_6, _pb_15, _pb_16, _pb_11, _pb_17, ones], axis=1)
    alpha = alpha_raw
    gain = _pb_2
    phi = _pb_4
    skew = _pb_5
    r7 = _pb_6
    relative = _pb_10
    alpha2 = _pb_11
    R = _pb_18
    hidden = fnp.tanh(R @ J['Wl'][layer] + fnp.mean(R, axis=0, keepdims=True) @ J['Ws'])
    h = fnp.tanh(hidden @ J['Wh_terminal']) * J['hs_terminal'] + J['ha_terminal'] if terminal else fnp.tanh(hidden @ J['Wh']) * J['hs'] + J['ha']
    return (h, sigma, hidden, alpha, relative, skew)

def read_geometry(runtime, hidden, P, K4, C, W, layer, gaussian, a, relative, skew, weight_squared):
    J = runtime.learned['joint']
    n = int(hidden.shape[0])
    (inv_s, inv_v) = (gaussian[6], gaussian[7])
    if P is None or C is None:
        extra = fnp.zeros((n, 12), dtype=fnp.float32)
    else:
        Z = fnp.einsum('ij,i,j->ij', P, inv_v, inv_s)
        fnp.fill_diagonal(Z, skew)
        Kn = fnp.einsum('ij,i,j->ij', K4, inv_v, inv_v)
        R = fnp.einsum('ij,i,j->ij', C, inv_s, inv_s)
        fnp.fill_diagonal(R, 0.0)
        rr = fnp.einsum('ij,ij->i', R, R)
        runtime.readout['rho'] = (C, R, rr)
        feats = fnp.stack([fnp.einsum('ij,ij->i', Z, Z), fnp.einsum('ij,ij->j', Z, Z), fnp.einsum('ij,ij->i', Kn, Kn), rr, fnp.einsum('ij,ji->i', Z, Z), fnp.einsum('ij,ij,ij->i', Kn, R, R), fnp.einsum('ij,ij->i', Z, R), fnp.einsum('ji,ij->i', Z, R), fnp.sum(Z, axis=1), fnp.sum(Z, axis=0), fnp.sum(Kn, axis=1), Z @ a], axis=1)
        den = fnp.sqrt(fnp.einsum('ij,ij->j', feats, feats) * (16.0 / float(n)))
        extra = fnp.tanh(feats / den)
    x = fnp.concatenate([hidden, extra, fnp.einsum('i,ij->ij', a, extra, optimize=False), fnp.einsum('i,ij->ij', relative, extra, optimize=False)], axis=1)
    t = fnp.tanh(x @ J['Wg'][layer])
    return t * J['dscale'] + J['dadd']

def load_birth_inputs(runtime, directory, width, depth):
    runtime.feature_layers = None
    if int(width) != 1024 or int(depth) != 16:
        return
    if directory is None:
        raise ValueError('fe31_inputs requires fe31_inputs.npz')
    from pathlib import Path as _P
    w = RegisteredWeights.from_file(str(_P(str(directory)) / 'fe31_inputs.npz'), group='fe31_inputs')
    if tuple(w.Wnew.shape) != (16, 11, 64) or tuple(w.mu.shape) != (16, 11) or tuple(w.sd.shape) != (16, 11):
        raise ValueError('FE31 weights must match the bound eleven-input contract')
    runtime.feature_layers = [(w.Wnew[l], w.mu[l], w.sd[l]) for l in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    runtime.learned['fe31_wnew'] = w.Wnew
    runtime.learned['fe31_mu'] = w.mu
    runtime.learned['fe31_sd'] = w.sd


def load_joint(runtime, directory):
    from pathlib import Path
    w = RegisteredWeights.from_file(str(Path(directory) / 'joint.npz'), group='joint')
    runtime.joint_weights = {k: getattr(w, v) for (k, v) in {'birth.bias': 'birth__bias', 'birth.weight': 'birth__weight', 'geometry_weight': 'geometry_weight', 'head.bias': 'head__bias', 'head.weight': 'head__weight', 'layer_bias': 'layer_bias', 'head_scale': 'head_scale'}.items()}

def apply_joint_geometry(state, frame):
    runtime = state.runtime
    layer = frame.layer
    weight = frame.weight
    frame.has_joint = layer >= 4
    if frame.has_joint:
        (frame.joint_h, frame.joint_sigma, joint_hidden, joint_alpha, joint_relative, joint_skew) = read_joint(runtime, frame.k3_diagonal, frame.k4_diagonal, layer, frame.wick, state.ones_vector, frame.gaussian, frame.is_final)
        frame.geometry_delta = read_geometry(runtime, joint_hidden, frame.k3_21, frame.k4_22, frame.covariance_pre, weight, layer, frame.gaussian, joint_alpha, joint_relative, joint_skew, frame.weight_squared)
        frame.joint_heads = fnp.unstack(frame.joint_h, axis=1)
        frame.k3_diagonal = frame.k3_diagonal * frame.joint_heads[0]
        if frame.k3_21 is not None:
            frame.k3_21 = frame.k3_21 * (frame.joint_heads[1][:, None] + frame.joint_heads[2][None, :])
        frame.k4_diagonal = frame.k4_diagonal * frame.joint_heads[3]
        if frame.k4_22 is not None:
            frame.k4_22 = frame.k4_22 * (frame.joint_heads[4][:, None] + frame.joint_heads[5][None, :])
        frame.geometry_heads = fnp.unstack(frame.geometry_delta, axis=1)
        (frame.k3_diagonal, frame.k4_diagonal) = fnp.unstack(fnp.polyval(
            [[frame.geometry_heads[0], frame.geometry_heads[1]], [frame.k3_diagonal, frame.k4_diagonal]],
            [frame.gaussian[9], frame.gaussian[8]]), axis=0)

def apply_corrector(state, frame):
    runtime = state.runtime
    layer = frame.layer
    weight = frame.weight
    _xf = None
    if frame.radial_birth_input is None and frame.radial_diagonal is not None and (state.k4_core is not None):
        frame.radial_birth_input = frame.radial_k4
    _xf = (frame.birth_diagonal, frame.radial_birth_input, frame.birth_slice, frame.birth_delta)
    if int(PARAMS['k21_slot']) and (not frame.is_final) and (layer in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) and (frame.k3_21 is not None):
        frame.k3_21 = apply_v2e(runtime, layer, frame.mean_pre, frame.variance_floor, frame.k3_diagonal, frame.k4_diagonal, frame.k3_21, frame.covariance_pre, frame.gaussian)
    frame.mean_delta = None
    (frame.mean_delta, state.corrector_carry) = correct_layer(runtime, layer, state.mlp.depth, frame.mean_pre, frame.k3_diagonal, frame.k4_diagonal, frame.covariance_pre, frame.k3_21, frame.k4_22, state.covariance, weight, state.corrector_carry, _xf, frame.gaussian, frame.weight_squared)

def correct_post_moments(state, frame):
    layer = frame.layer
    if layer < 4 or not frame.has_joint:
        return
    frame.k2_diagonal = frame.k2_diagonal * frame.joint_heads[7]
    frame.k2_offdiag = frame.k2_offdiag * (frame.joint_heads[8][:, None] + frame.joint_heads[9][None, :])
    frame.k2_diagonal = frame.k2_diagonal * frame.geometry_heads[2]
    frame.k2_offdiag = frame.k2_offdiag * (frame.geometry_heads[3][:, None] + frame.geometry_heads[4][None, :])
