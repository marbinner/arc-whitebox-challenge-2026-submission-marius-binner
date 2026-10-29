"""Named R8 inputs, shapes, and conservative compute bounds.

Ratio features provide numerators/denominators and share one fused stack/divide.
Matrix features follow that stack. The default preserves R1's operation order.
Custom functions accept FeatureContext; register a width and a cost with them.
"""
from __future__ import annotations
from types import SimpleNamespace
from .state import flops, fnp
from .corrector import centred_weight_squared, standardized_birth_inputs

class FeatureContext(SimpleNamespace):
    """Explicit runtime/frame values used by one corrector evaluation."""

def feature_marginals(ctx):
    ones = ctx.runtime.readout['ones']
    (v, s, a, Phi, a2, phi, inv_s, inv_v, v2, vs) = ctx.gaussian
    W2 = ctx.W * ctx.W if ctx.weight_squared is None else ctx.weight_squared
    num = [a, Phi, phi, s / fnp.mean(s), v / fnp.mean(v), ctx.k3, ctx.k4, fnp.sum(W2, axis=0), fnp.einsum('ij,i->j', W2, fnp.diagonal(ctx.C_prev))]
    den = [ones, ones, ones, ones, ones, vs, v2, ones, v]
    return num, den


def feature_pairs(ctx):
    runtime, C, P, K = ctx.runtime, ctx.C, ctx.P, ctx.K
    ones = runtime.readout['ones']
    (v, s, a, Phi, a2, phi, inv_s, inv_v, v2, vs) = ctx.gaussian
    num, den = [], []
    rho = None
    if C is not None:
        shared = runtime.readout.get('rho')
        if shared is not None and shared[0] is C:
            (rho, rr) = (shared[1], shared[2])
        else:
            rho = fnp.einsum('ij,i,j->ij', C, inv_s, inv_s)
            fnp.fill_diagonal(rho, 0.0)
            rr = fnp.einsum('ij,ij->i', rho, rho)
        if P is None:
            zero = runtime.readout.get('zero')
            if zero is None:
                zero = fnp.zeros((int(v.shape[0]),), dtype=fnp.float32)
                runtime.readout['zero'] = zero
            pn_features = [zero] * 5
        else:
            pn = fnp.einsum('ij,i,j->ij', P, inv_v, inv_s)
            fnp.fill_diagonal(pn, 0.0)
            pn_features = [fnp.einsum('ij,ij->i', pn, pn), fnp.einsum('ij,ij->j', pn, pn), fnp.einsum('ij,ij->i', pn, rho), fnp.sum(pn, axis=1), fnp.sum(pn, axis=0)]
        num += [rr, fnp.einsum('ij,ij,ij->i', rho, rho, rho), rho @ (Phi * 2.0 - 1.0)] + pn_features
        if K is not None:
            kn = fnp.einsum('ij,i,j->ij', K, inv_v, inv_v)
            fnp.fill_diagonal(kn, 0.0)
            num += [fnp.sum(kn, axis=1), fnp.einsum('ij,ij->i', kn, kn)]
        else:
            zero = runtime.readout.get('zero')
            if zero is None:
                zero = fnp.zeros((int(v.shape[0]),), dtype=fnp.float32)
                runtime.readout['zero'] = zero
            num += [zero, zero]
        den += [ones] * 10
    return num, den


def feature_physics(ctx):
    runtime, layer, xf = ctx.runtime, ctx.layer, ctx.xf
    w = runtime.corrector_layers[layer]
    ones = runtime.readout['ones']
    (v, s, a, Phi, a2, phi, inv_s, inv_v, v2, vs) = ctx.gaussian
    num, den = [], []
    if 'Wxf' in w:
        (xD, xR, xY, xK) = (None, None, None, None) if xf is None else xf
        if xD is None or xR is None or xK is None:
            zero = runtime.readout.get('zero')
            if zero is None:
                zero = fnp.zeros((int(v.shape[0]),), dtype=fnp.float32)
                runtime.readout['zero'] = zero
        num += [zero if xD is None else xD, zero if xR is None else xR]
        den += [v2, v2]
        num.append(zero if xK is None else xK)
        den.append(v2)
    return num, den


def feature_bias(ctx):
    ones = ctx.runtime.readout['ones']
    return [ones], [ones]


def feature_carry(ctx):
    runtime, W, carry = ctx.runtime, ctx.W, ctx.carry
    W2 = W * W if ctx.weight_squared is None else ctx.weight_squared
    mem = None
    if carry is not None:
        (prev, Phi_h, T_p, mem_p) = carry
        wp = runtime.corrector_layers[prev]
        hp = Phi_h[:, None]
        Whp = W * hp
        right = fnp.einsum('ij,bjk->bik', T_p, runtime.learned['corr'][prev]['ABC3'])
        left = fnp.stack([Whp, centred_weight_squared(runtime, W2, int(W.shape[0])), W2 * W], axis=0)
        msg = fnp.einsum('bij,bik->jk', left, right)
        t1 = fnp.tanh(msg)
        gate = fnp.tanh(msg @ wp['Gz'] + wp['bz'])
        if mem_p is None:
            mem = t1 + gate * t1
        else:
            t2 = fnp.tanh(fnp.einsum('ij,ik->jk', Whp, mem_p))
            mem = t1 + t2 + gate * (t1 - t2)
    return mem


def feature_birth_inputs(ctx):
    v = ctx.gaussian[0]
    return standardized_birth_inputs(ctx.runtime, ctx.layer, int(v.shape[0]), v) if ctx.layer != 0 and ctx.runtime.feature_layers is not None else None


FEATURE_REGISTRY = {
    'marginals': dict(function=feature_marginals, **{'width': 9, 'shape': '(width, 9)', 'kind': 'ratio', 'cost': {'flops_upper': '3*n*n + 64*n', 'requests_upper': 24}, 'requires': ['gaussian', 'k3_diagonal', 'k4_diagonal', 'weight_squared', 'previous_covariance'], 'owner': 'corrector', 'doc': 'R8 marginals input group; absent structural inputs contribute zero columns.'}),
    'pairs': dict(function=feature_pairs, **{'width': 10, 'shape': '(width, 10)', 'kind': 'ratio', 'cost': {'flops_upper': '40*n*n + 8*n', 'requests_upper': 48}, 'requires': ['covariance_pre', 'k3_21', 'k4_22', 'gaussian'], 'owner': 'corrector', 'doc': 'R8 pairs input group; absent structural inputs contribute zero columns.'}),
    'physics': dict(function=feature_physics, **{'width': 3, 'shape': '(width, 3)', 'kind': 'ratio', 'cost': {'flops_upper': '3*n', 'requests_upper': 4}, 'requires': ['birth_diagonal', 'radial_birth_input', 'birth_delta', 'gaussian'], 'owner': 'corrector', 'doc': 'R8 physics input group; absent structural inputs contribute zero columns.'}),
    'bias': dict(function=feature_bias, **{'width': 1, 'shape': '(width, 1)', 'kind': 'ratio', 'cost': {'flops_upper': '0', 'requests_upper': 0}, 'requires': ['ones'], 'owner': 'corrector', 'doc': 'R8 bias input group; absent structural inputs contribute zero columns.'}),
    'carry': dict(function=feature_carry, **{'width': 16, 'shape': '(width, 16)', 'kind': 'matrix', 'cost': {'flops_upper': '256*n*n + 4096*n', 'requests_upper': 64}, 'requires': ['corrector_carry', 'weight', 'weight_squared', 'corrector_assets'], 'owner': 'corrector', 'doc': 'R8 carry input group; absent structural inputs contribute zero columns.'}),
    'birth_inputs': dict(function=feature_birth_inputs, **{'width': 11, 'shape': '(width, 11)', 'kind': 'matrix', 'cost': {'flops_upper': '160*n', 'requests_upper': 28}, 'requires': ['runtime.features', 'fe31_inputs', 'gaussian'], 'owner': 'corrector', 'doc': 'R8 birth inputs input group; absent structural inputs contribute zero columns.'}),
}

DEFAULT_FEATURES = ('marginals', 'pairs', 'physics', 'bias', 'carry', 'birth_inputs')


def validate_features(names):
    names = tuple(names)
    if len(names) != len(set(names)) or set(names) - set(FEATURE_REGISTRY):
        raise ValueError('Unknown or duplicate corrector feature')
    required = {'function', 'width', 'shape', 'kind', 'cost', 'requires'}
    for name in names:
        spec = FEATURE_REGISTRY[name]
        if required - set(spec) or not callable(spec['function']):
            raise ValueError('Incomplete feature contract: ' + name)
        if not isinstance(spec['width'], int) or spec['width'] < 1 or spec['kind'] not in ('ratio', 'matrix'):
            raise ValueError('Invalid feature shape/kind: ' + name)
        if {'flops_upper', 'requests_upper'} - set(spec['cost']):
            raise ValueError('Feature must declare both costs: ' + name)
    return names


def feature_width(names):
    return sum(FEATURE_REGISTRY[name]['width'] for name in names)


def feature_layout(names):
    """Canonical row coordinates for exported linear/input projection weights."""
    names = validate_features(names)
    ordered = [name for kind in ('ratio', 'matrix') for name in names
               if FEATURE_REGISTRY[name]['kind'] == kind]
    layout, offset = [], 0
    for name in ordered:
        width = FEATURE_REGISTRY[name]['width']
        layout.append(dict(name=name, start=offset, stop=offset + width, kind=FEATURE_REGISTRY[name]['kind']))
        offset += width
    return layout


def build_corrector_features(ctx, names):
    names = validate_features(names)
    numerator, denominator, columns = [], [], []
    offsets, offset = {}, 0
    ordered = [name for name in names if FEATURE_REGISTRY[name]['kind'] == 'ratio']
    ordered += [name for name in names if FEATURE_REGISTRY[name]['kind'] == 'matrix']
    n = int(ctx.gaussian[0].shape[0])
    for name in ordered:
        offsets[name] = offset
        offset += FEATURE_REGISTRY[name]['width']
    for name in ordered:
        spec = FEATURE_REGISTRY[name]
        if spec['kind'] == 'ratio':
            num, den = spec['function'](ctx)
            if len(num) != len(den) or len(num) > spec['width']:
                raise ValueError('Feature ratio width mismatch: ' + name)
            if any(tuple(v.shape) != (n,) for v in tuple(num) + tuple(den)):
                raise ValueError('Feature ratio row shape mismatch: ' + name)
            numerator.extend(num)
            denominator.extend(den)
            columns.extend(range(offsets[name], offsets[name] + len(num)))
    F = (fnp.stack(numerator, axis=1) / fnp.stack(denominator, axis=1)) if numerator else fnp.zeros((n, 0), dtype=fnp.float32)
    parts, matrices = [F], {}
    for name in ordered:
        spec = FEATURE_REGISTRY[name]
        if spec['kind'] == 'matrix':
            value = spec['function'](ctx)
            matrices[name] = value
            if value is not None:
                if tuple(value.shape) != (n, spec['width']):
                    raise ValueError('Feature matrix width mismatch: ' + name)
                parts.append(value)
                columns.extend(range(offsets[name], offsets[name] + spec['width']))
    return dict(parts=parts, matrices=matrices, names=tuple(ordered), columns=columns,
                total_width=offset, context=ctx)
