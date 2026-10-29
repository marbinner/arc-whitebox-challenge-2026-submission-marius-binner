"""f3 quadratic corrector parameterization plus registry-owned knob/asset leaves."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .folding import DEPTH, fold_corrector, layer_weights
from .runtime import cpu_tree, literal
from .selection import registry_entries, select

# f3 kernel_v172SgD's unscaled K31 table, used only to import legacy band knobs.
K31_REFERENCE = (0.9348154514941799, 0.867516989294848, 0.7978633686903603)
PLAIN_KNOBS = ('split_pair_c3_cov', 'k4_off_scale', 'edgeworth_34', 'cov_k3sq',
               'cov_k3cross', 'k3_vertex_dress', 'cov_adt3', 'pair_ck3',
               'k4_22_dress', 'h12_fold_gain', 'k4_d31c_scale')


def parameter_tree(value, device):
    if isinstance(value, dict):
        return {key: parameter_tree(item, device) for key, item in value.items()}
    return torch.as_tensor(value, dtype=torch.float64, device=device).detach().clone()


def tree_tensors(value, prefix):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from tree_tensors(item, prefix + '.' + str(key))
    else:
        yield prefix, value


def tensor_request(value):
    """Float32 differentiable views; float64 master leaves stay in the optimizer."""
    if isinstance(value,dict):
        return {key:tensor_request(item) for key,item in value.items()}
    return value.float()


def read_norms(norms):
    if not norms:
        raise ValueError('Legacy corrector checkpoints require --norms')
    if isinstance(norms, dict):
        return norms
    if str(norms).endswith('.npz'):
        with np.load(norms, allow_pickle=False) as z:
            return dict(row_norm=[torch.as_tensor(z['row_norm_mean']), torch.as_tensor(z['row_norm_std'])],
                        xf_norm=[torch.as_tensor(z['xf_norm_mean']), torch.as_tensor(z['xf_norm_std'])]
                        if 'xf_norm_mean' in z.files else None)
    return torch.load(norms, map_location='cpu', weights_only=False)


def legacy_parameters(checkpoint, defaults):
    """f3 export mapping, including MB1's three newly restored PARAMS keys."""
    values = copy.deepcopy(defaults)
    knobs = checkpoint.get('knobs') or {}
    r9 = lambda value: round(float(value), 9)
    for key in PLAIN_KNOBS:
        if key in knobs and key in values:
            values[key] = r9(knobs[key])
    for key in ('c_Yi', 'c_Yii'):
        if key + '_g1' in knobs and key in values:
            values[key] = (values[key][0], r9(knobs[key + '_g1']), r9(knobs[key + '_g2']))
    if 'k31b_b0' in knobs and 'exact_k31_betas' in values:
        values['exact_k31_betas'] = {layer: r9(K31_REFERENCE[0 if int(layer) <= 5 else 1 if int(layer) <= 10 else 2] *
             knobs['k31b_b' + str(0 if int(layer) <= 5 else 1 if int(layer) <= 10 else 2)])
             for layer in values['exact_k31_betas']}
    if 'fe30_c_b1' in knobs and 'fe30' in values:
        values['fe30']['coef'] = [float(knobs['fe30_c_b1']), float(knobs['fe30_c_b2'])]
    if 'inner_k4_b0' in knobs and 'fe34' in values:
        values['fe34']['inner_k4'] = [r9(knobs['inner_k4_b' + str(b)]) for b in range(3)]
    if checkpoint.get('beta') is not None:
        values['feed_fold_coeffs'] = np.round(checkpoint['beta'].cpu().double().numpy(), 6).tolist()
    if checkpoint.get('k3band') is not None:
        values['r_k3band'] = np.round(checkpoint['k3band'].cpu().double().numpy(), 9).tolist()
    if checkpoint.get('v2e_state') and 'gate' in checkpoint['v2e_state']:
        gates = list(values['k21_slot_v2e_gates'])
        for index, layer in enumerate(range(4, 15)):
            gates[layer] = float(checkpoint['v2e_state']['gate'][index])
        values['k21_slot_v2e_gates'] = gates
    values.update(checkpoint.get('params_all') or {})
    return values


def _differs(a, b):
    try:
        a = np.asarray(literal(a), dtype=float); b = np.asarray(literal(b), dtype=float)
        return a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-12)
    except (TypeError, ValueError):
        return literal(a) != literal(b)


def source_wins(checkpoint_values, source_values, explicit, label, previous_source=None):
    """Warm starts keep the checkpoint's knob values; a source package whose registry defaults changed (new physics)
    is silently reverted. Report every such key. WHEST_INIT_PARAMS=source makes every differing source default win,
    =checkpoint keeps every checkpoint value. Unset/auto: when the checkpoint records the source defaults it started
    from (previous_source), a key whose source default CHANGED since then takes the new default (new physics) and a
    key whose default is unchanged keeps the checkpoint's trained value; older checkpoints keep the checkpoint value."""
    import os, sys
    changed = [key for key, value in source_values.items()
               if key not in explicit and key in checkpoint_values and _differs(checkpoint_values[key], value)]
    if not changed:
        return {}
    mode = os.environ.get('WHEST_INIT_PARAMS', 'auto')
    if mode == 'source':
        wins = list(changed)
    elif mode == 'checkpoint' or previous_source is None:
        wins = []
    else:
        wins = [key for key in changed if key not in previous_source or _differs(previous_source[key], source_values[key])]
    print(json.dumps(dict(warning='warm start: checkpoint knob values differ from source defaults', keys=changed,
                          used=mode, source_defaults_taken=wins, label=label)), file=sys.stderr, flush=True)
    return {key: source_values[key] for key in wins}


class TrainState:
    def __init__(self, twin, checkpoint=None, norms=None, *, names=(), groups=(), owners=(), per_layer=(),warm_start=False,
                 freeze_zero_knobs=False):
        self.device, self.schema = twin.device, twin.schema
        self.defaults = {key: copy.deepcopy(spec['default']) for key, spec in self.schema['parameters'].items()}
        self.source_params = copy.deepcopy(getattr(twin,'source_params',
            {**self.defaults,**getattr(twin,'source_config',twin.config).get('params',{})}))
        self.base_assets = {group: {key: value.detach().clone() for key, value in members.items()}
                            for group, members in twin.runtime_assets().items()}
        self.theta, self.v2e, self.wnew, self.row_norm, self.xf_norm = {}, {}, None, None, None
        self.extra_assets, self.grown = {}, {}
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False) if isinstance(checkpoint, (str, Path)) else checkpoint
        from .data import checkpoint_provenance,package_provenance,merge_provenance
        source_provenance=package_provenance(twin.base)
        self.training_provenance=(checkpoint_provenance(checkpoint) if checkpoint is not None
                                  else source_provenance)
        if checkpoint is not None and source_provenance['seen_indices']:
            self.training_provenance=merge_provenance(self.training_provenance,source_provenance)
        continuation = checkpoint.get('train_state') if checkpoint and checkpoint.get('schema') == 'whest-train-v1' else None
        if continuation:
            # The checkpoint's recorded source defaults (absent in older checkpoints) let a warm start tell trained
            # knobs (same default) from new physics (changed default); the active source's values stay current.
            previous_source = continuation.get('source_params')
            self.restore({key: value for key, value in continuation.items() if key != 'source_params'})
            self.selected = select(self.schema, names=names, groups=groups, owners=owners) if names or groups or owners or warm_start else list(continuation['selected'])
            if warm_start:
                for key, value in source_wins(literal(self.params), self.source_params,
                                              set(twin.config.get('params', {})), 'v1 warm start',
                                              previous_source=previous_source).items():
                    self.params[key] = parameter_tree(twin.normalize_parameter(key, value), self.device)
                for key,spec in self.schema['parameters'].items():
                    if spec['kind']=='continuous' and (key not in self.params or key in twin.config.get('params',{})):
                        value=twin.config.get('params',{}).get(key,self.defaults[key])
                        self.params[key]=parameter_tree(twin.normalize_parameter(key,value),self.device)
                if 'per_layer' in twin.config:
                    self.schedules={key:parameter_tree(value,self.device) for key,value in twin.config['per_layer'].items()}
                for group,members in twin.runtime_assets().items():
                    stored=self.base_assets.setdefault(group,{})
                    for key,value in members.items():
                        if key not in stored: stored[key]=value.detach().clone()
        elif checkpoint and 'corr_theta' in checkpoint:
            n = read_norms(norms)
            move = lambda tensor: tensor.detach().clone().to(self.device)
            self.theta = {key: move(value).float() for key, value in checkpoint['corr_theta'].items()}
            self.row_norm = [move(value) for value in n['row_norm']]
            self.xf_norm = None if n.get('xf_norm') is None else [move(value) for value in n['xf_norm']]
            self.v2e = {key: move(value).float() for key, value in checkpoint['v2e_state'].items() if key != 'gate'}
            self.wnew = None if checkpoint.get('fe31') is None else move(checkpoint['fe31']['wnew']).float()
            values = legacy_parameters(checkpoint, self.defaults)
            values.update(source_wins(values, self.source_params, set(twin.config.get('params', {})), 'legacy init'))
            values.update(twin.config.get('params',{}))
            self.params = {key: parameter_tree(twin.normalize_parameter(key, value), self.device)
                           for key, value in values.items() if self.schema['parameters'][key]['kind'] == 'continuous'}
            self.schedules = {}
            self.selected = select(self.schema, names=names, groups=groups, owners=owners)
        else:
            values={**self.source_params,**twin.config.get('params',{})}
            self.params = {key: parameter_tree(twin.normalize_parameter(key,value), self.device) for key, value in values.items()
                           if self.schema['parameters'][key]['kind'] == 'continuous'}
            self.schedules = {}
            self.selected = select(self.schema, names=names, groups=groups, owners=owners)
        if not continuation:
            self.schedules={key:parameter_tree(value,self.device) for key,value in twin.config.get('per_layer',{}).items()}
        for key in per_layer:
            twin.expand_scalar_parameter(key)
            if 'params.' + key not in self.selected:
                raise ValueError('Per-layer parameter must be selected: ' + key)
            if key not in self.schedules:
                self.schedules[key] = self.params[key].detach().repeat(16).clone()
        if freeze_zero_knobs:
            # A knob at exactly 0 switches its branch off in the native (enabled(v) is v != 0); Adam's first noisy
            # steps would switch it back on and the export would pay that branch's FLOPs and requests (cov_adt3).
            import sys
            zero = sorted(name for name in self.selected if name.startswith('params.') and name[7:] in self.params
                          and name[7:] not in self.schedules
                          and not any(bool(torch.any(value != 0)) for _, value in tree_tensors(self.params[name[7:]], name)))
            if zero:
                print(json.dumps(dict(warning='zero-valued knobs stay frozen; --train-zero-knobs trains them', keys=zero)),
                      file=sys.stderr, flush=True)
                self.selected = [name for name in self.selected if name not in zero]
        self.configure_leaves()

    def configure_leaves(self):
        """Map named registered folded tensors onto their f3 source leaves."""
        selected = set(self.selected)
        self.leaves, self.leaf_names = {}, {}
        def add(name, tensor, registry_names):
            active = bool(selected.intersection(registry_names))
            tensor.requires_grad_(active)
            if active:
                self.leaves[name], self.leaf_names[name] = tensor, sorted(selected.intersection(registry_names))
        corr_map = {'Win': ('Wx','Wxr','Wxm','Wxrm','Wxf','bin'), 'bin': ('bin',),
                    'Wout': ('Wom',), 'Wm': ('Wom',), 'bout': ('bom',), 'bm': ('bom',),
                    'A': ('ABC',), 'B': ('ABC',), 'C3': ('ABC',)}
        for key, value in self.theta.items():
            add('corr.' + key, value, ['corrector.' + member for member in corr_map.get(key, (key,))])
        v2e_map = {'f1.weight': ('W0','b0'), 'f1.bias': ('b0',), 'f2.weight': ('W1',),
                   'f2.bias': ('b1',), 'out.weight': ('Wout',), 'out.bias': ('bout',)}
        for key, value in self.v2e.items():
            add('v2e.' + key, value, ['k21_v2e.' + member for member in v2e_map[key]])
        if self.wnew is not None:
            add('fe31.wnew', self.wnew, ['fe31_inputs.Wnew'])
        for key, tree in self.params.items():
            for name, value in tree_tensors(tree, 'params.' + key):
                add(name, value, [] if key in self.schedules else ['params.' + key])
        for key, value in self.schedules.items():
            add('per_layer.' + key, value, ['params.' + key])
        handled = {'corrector'} if self.theta else set()
        if self.v2e: handled.add('k21_v2e')
        for name in self.selected:
            if name.startswith('params.'): continue
            group, key = name.split('.', 1)
            if group in handled or group == 'fe31_inputs' and self.wnew is not None:
                continue
            if group not in self.base_assets or key not in self.base_assets[group]:
                raise ValueError('Selected registered asset is absent from this runtime: ' + name)
            self.extra_assets.setdefault(group, {})
            if key not in self.extra_assets[group]:
                self.extra_assets[group][key] = self.base_assets[group][key].detach().clone()
            add(name, self.extra_assets[group][key], [name])
        if not self.leaves:
            raise ValueError('No optimizer leaves remain after registry selection')

    def request(self, *, evaluation=False):
        # Scalar contractions promote float64 coefficient tensors, unlike
        # native Python scalar operands. Keep the native model's float32
        # tensor arithmetic during training; literal eval/export preserves
        # its exact float64-to-float32 setup rounding. Casts retain gradients.
        params = {key:literal(value) if evaluation else tensor_request(value)
                  for key,value in self.params.items()}
        assets = {group: dict(members) for group, members in self.extra_assets.items()}
        if self.theta:
            assets['corrector'] = fold_corrector(self.theta, self.row_norm, self.xf_norm)
        if self.wnew is not None:
            assets['fe31_inputs'] = {'Wnew': torch.stack([layer_weights({'X': self.wnew.double()}, layer)['X'] for layer in range(DEPTH)]).float()}
        if self.v2e:
            s = self.v2e; b = self.base_assets['k21_v2e']; b0 = list(b['b0'].unbind())
            for layer in range(4, 15):
                b0[layer] = (s['f1.bias'] + s['f1.weight'][:, 17 + layer - 4]).float()
            assets['k21_v2e'] = dict(W0=s['f1.weight'][:, :17].T.float(), b0=torch.stack(b0).float(),
                 W1=s['f2.weight'].T.float(), b1=s['f2.bias'].float(), Wout=s['out.weight'].T.float(), bout=s['out.bias'].float())
        request = dict(params=params, assets=assets)
        if self.theta:
            ff=int(self.theta['Wf0'].shape[-1])
            blocks=sum(key.startswith('Wf') for key in self.theta)
            if ff!=128 or blocks!=2:
                request['corrector']=dict(ff_width=ff,blocks=blocks)
        if self.schedules:
            request['per_layer'] = literal(self.schedules) if evaluation else tensor_request(self.schedules)
        return request

    def refresh_schema(self,twin,*,include_new_corrector=False):
        """Accept the public schema after an atomic structural asset update."""
        previous={name for name,spec in self.schema['assets'].items()
                  if name.startswith('corrector.') and spec.get('trainable') and spec.get('kind')=='continuous'}
        schema=twin.model.runtime_schema()
        if include_new_corrector and previous.issubset(self.selected):
            for name,spec in schema['assets'].items():
                if name.startswith('corrector.') and spec.get('trainable') and spec.get('kind')=='continuous' and name not in self.selected:
                    self.selected.append(name)
        self.selected=select(schema,names=self.selected)
        self.schema=twin.schema=schema
        self.configure_leaves()

    def optimizer_groups(self, *, lr_corr=1e-5, lr_v2e=3e-4, lr_knobs=1e-3,
                         lr_beta=5e-4, lr_fe31=3e-4, lr_k3band=1e-4,
                         lr_scale=1.0, overrides=None, grow_lr=1e-4, lr_joint=0.0, knob_floor=0.3):
        overrides = overrides or {}
        valid=set(self.leaves)|set(self.selected)|{name.removeprefix('params.') for name in self.selected}
        if set(overrides)-valid:
            raise ValueError('Learning-rate override is not a selected trainable: '+', '.join(sorted(set(overrides)-valid)))
        result = []
        for name, tensor in self.leaves.items():
            if name.startswith('corr.') or name.startswith('corrector.'):
                key = name.split('.', 1)[1]
                lr = grow_lr if key in self.grown else lr_corr
            elif name.startswith(('v2e.', 'k21_v2e.')): lr = lr_v2e
            elif name.startswith(('fe31.', 'fe31_inputs.')): lr = lr_fe31
            # Joint/geometry heads: own rate when lr_joint > 0; 0 keeps the legacy lr_corr fallback.
            elif name.startswith('joint.') and lr_joint > 0: lr = lr_joint
            elif name.startswith(('params.', 'per_layer.')):
                key = name.split('.')[1]
                if key == 'feed_fold_coeffs': lr = lr_beta
                elif key == 'r_k3band': lr = lr_k3band
                elif key == 'k21_slot_v2e_gates': lr = lr_v2e
                else:
                    magnitude = float(tensor.detach().abs().mean())
                    # Epsilon knobs need relative steps; f3's 0.3 floor fits
                    # the physical scale knobs, but destroys tiny epsilons.
                    # --knob-floor lowers it for small knobs (r8_output_scale 0.005, leg_tail_energy 0.002,
                    # k3_carrier_mu_var 0.007, k4_vertex_dress 0.01 move 3-15 %/step at the 0.3 floor).
                    scale = max(magnitude, 1e-12) if 'epsilon' in key else max(magnitude, knob_floor)
                    lr = lr_knobs * scale
            else: lr = lr_corr
            for registry_name in self.leaf_names[name]:
                lr = overrides.get(registry_name, overrides.get(registry_name.removeprefix('params.'), lr))
            lr = overrides.get(name, lr) * lr_scale
            if not math.isfinite(lr) or lr < 0: raise ValueError('Learning rates must be finite and nonnegative')
            group=dict(params=[tensor],lr=lr,base_lr=lr,name=name)
            if name.startswith(('corr.','corrector.')) and key in self.grown:
                explicit=any(item in overrides for item in [name,*self.leaf_names[name]])
                if grow_lr<=0: raise ValueError('Growth learning rate must be positive')
                group['old_slice_ratio']=1.0 if explicit else lr_corr/grow_lr
            result.append(group)
        return result

    def clamp_parameters(self):
        with torch.no_grad():
            for key, tree in self.params.items():
                bounds = self.schema['parameters'][key].get('bounds')
                if bounds:
                    for _, tensor in tree_tensors(tree, key): tensor.clamp_(*bounds)
            for key, tensor in self.schedules.items():
                bounds = self.schema['parameters'][key].get('bounds')
                if bounds: tensor.clamp_(*bounds)

    def snapshot(self):
        return cpu_tree(dict(theta=self.theta, v2e=self.v2e, wnew=self.wnew,
                    row_norm=self.row_norm, xf_norm=self.xf_norm, params=self.params,
                    schedules=self.schedules, base_assets=self.base_assets,
                    extra_assets=self.extra_assets, selected=self.selected, grown=self.grown,
                    source_params=copy.deepcopy(literal(self.source_params))))

    def restore(self, value):
        def move(item):
            if isinstance(item, torch.Tensor): return item.detach().clone().to(self.device)
            if isinstance(item, dict): return {k: move(v) for k,v in item.items()}
            if isinstance(item, (tuple,list)): return type(item)(move(v) for v in item)
            return copy.deepcopy(item)
        for key, item in value.items():
            setattr(self, key, move(item))

    def grow(self, ff=None, nb=None, seed=7, init='zero'):
        if not self.theta:
            raise ValueError('--grow requires a legacy quadratic corrector checkpoint')
        if init not in ('zero', 'split'):
            raise ValueError('Growth init must be zero or split')
        th = self.theta; gen = torch.Generator(device='cpu').manual_seed(seed)
        nb0 = sum(key.startswith('Wf') for key in th); dim, ff0 = th['Wf0'].shape[1:]
        if ff is not None and ff<ff0 or nb is not None and nb<nb0:
            raise ValueError('--grow cannot shrink the corrector')
        ff, nb = max(int(ff or ff0), ff0), max(int(nb or nb0), nb0)
        if init == 'split' and ff > 2 * ff0:
            raise ValueError('Split growth copies each unit at most once (ff <= 2x the current width)')
        z = lambda *shape: torch.zeros(shape, device=self.device, dtype=th['Wf0'].dtype)
        r = lambda *shape: (torch.randn(shape, generator=gen) * 0.02).to(self.device, th['Wf0'].dtype)
        old = {key: tuple(value.shape) for key, value in th.items()}
        for block in range(nb0):
            if ff > ff0 and init == 'split':
                # Net2Wider: each new unit copies an existing unit's input weights and bias (largest output norm first);
                # the pair shares that unit's output weights as Wg/2 + eta and Wg/2 - eta. Both units see the same
                # activation, so the block's function is unchanged, while eta gives the copies different gradients. Zero
                # output init instead leaves new units near 0 for hundreds of noisy updates.
                wf, bf, wg = (th[f'{name}{block}'].detach() for name in ('Wf', 'bf', 'Wg'))
                source = wg.double().pow(2).sum((0, 2)).argsort(descending=True)[:ff - ff0].to(wg.device)
                half = wg[:, source, :] / 2
                eta = half * (torch.rand(half.shape, generator=gen) * 0.2 - 0.1).to(self.device, wg.dtype)
                kept = wg.clone(); kept[:, source, :] = half + eta
                th[f'Wf{block}'] = torch.cat([wf, wf[:, :, source]], 2)
                th[f'bf{block}'] = torch.cat([bf, bf[:, source]], 1)
                th[f'Wg{block}'] = torch.cat([kept, half - eta], 1)
            elif ff > ff0:
                pad = z(3, dim, ff - ff0); pad[0] = r(dim, ff - ff0)
                th[f'Wf{block}'] = torch.cat([th[f'Wf{block}'].detach(), pad], 2)
                th[f'bf{block}'] = torch.cat([th[f'bf{block}'].detach(), z(3, ff-ff0)], 1)
                th[f'Wg{block}'] = torch.cat([th[f'Wg{block}'].detach(), z(3, ff-ff0, dim)], 1)
        for block in range(nb0, nb):
            wf = z(3,dim,ff); wf[0] = r(dim,ff)
            th.update({f'Wf{block}':wf, f'bf{block}':z(3,ff), f'Wg{block}':z(3,ff,dim),
                       f'bg{block}':z(3,dim), f'Wc{block}':z(3,dim,dim)})
        self.grown.update({key:old.get(key) for key,value in th.items() if old.get(key) != tuple(value.shape)})
        self.configure_leaves()
        return self.grown
