"""Fourth-cumulant radial host, carried stars, pair dressing and k31 fold.
Math: diagonal=h*r**2+centered stars; host uses trace and radial Frobenius terms.
Inputs: host/source vertices and leg products. Outputs: diagonal/22 cumulants,
birth products and next host. Knobs: registered k4 weights and exact_k31_betas."""
from __future__ import annotations
from .array_ops import elementwise_stack, zero_diagonal_owned
from .config import PARAMS
from .state import VAR_FLOOR, fnp

def k31_first_product(runtime, t3, tq2, t2q, c31, star, c22):
    return fnp.einsum('...ij,...j->...ij', t3, c31, optimize=False) + fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['three'], star, tq2, optimize=False) + fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['three'], c22, t2q, optimize=False)

def k31_second_product(runtime, t3, t2q, q2, q, tq2, c4, c31, star, c22):
    return fnp.einsum('...ij,...j->...ij', t3, c4, optimize=False) + fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['three'], c31, t2q, optimize=False) + fnp.einsum('gij,gij,gj->gij', q2, q, star, optimize=False) + fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['three'], c22, tq2, optimize=False)

def dress_k4_pair(k4_22, addition, radial, dress):
    ratio = addition / radial
    cross = fnp.outer(ratio, radial)
    return zero_diagonal_owned(k4_22 + dress / 6.0 * (cross + cross.T))

def reduce_k4_stars(state, group_indices, suffix_parts, suffix_squared, repeated_squared, repeated_suffix, star_channel, family_channel, column_width):
    star_vertex_batch = vertex_22_batch = vertex_31_batch = vertex_4o_batch = None
    if state.source_k4_star_vertices is not None:
        star_vertex_batch = elementwise_stack([state.source_k4_star_vertices[index] for index in group_indices], column_width, 0, cache=state.source_padding)
        _av_k4_t2 = suffix_squared
        _av_k4_q2 = repeated_squared
        leg_tq = repeated_suffix
        if state.source_k4_family_vertices is not None and (state.k4_family_ratios[0] != 0.0 or state.k4_family_ratios[2] != 0.0):
            vertex_22_batch = elementwise_stack([state.source_k4_family_vertices[index][0] for index in group_indices], column_width, 0, cache=state.source_padding)
            star_terms = leg_tq * (fnp.einsum('gij,gj->gij', _av_k4_q2, star_vertex_batch, optimize=False) + fnp.einsum('gij,gj->gij', leg_tq, vertex_22_batch, optimize=False))
        else:
            star_terms = fnp.einsum('gij,gj->gij', leg_tq * _av_k4_q2, star_vertex_batch, optimize=False)
        group_star = fnp.sum(fnp.sum(star_terms, axis=2), axis=0)
        star_channel = group_star if star_channel is None else star_channel + group_star
        if state.source_k4_family_vertices is not None:
            (ratio_22c, ratio_31c, ratio_22g) = state.k4_family_ratios
            group_family = None
            if ratio_31c != 0.0:
                vertex_31_batch = elementwise_stack([state.source_k4_family_vertices[index][1] for index in group_indices], column_width, 0, cache=state.source_padding)
                channel_31 = fnp.einsum('gij,gij,gj->i', _av_k4_t2, leg_tq, vertex_31_batch, optimize=False)
                group_family = channel_31 if group_family is None else group_family + channel_31
            if float(PARAMS['k4_d40o_scale']) != 0.0:
                vertex_4o_batch = elementwise_stack([state.source_k4_family_vertices[index][2] for index in group_indices], column_width, 0, cache=state.source_padding)
            if group_family is not None:
                family_channel = group_family if family_channel is None else family_channel + group_family
    return (star_channel, family_channel, star_vertex_batch, vertex_22_batch, vertex_31_batch, vertex_4o_batch)

def k4_birth_normalization_needed(runtime):
    for key in ('k4_star_scale', 'k4_d22c_scale', 'k4_d31c_scale'):
        values = runtime.per_layer.get(key, ())
        if len(values) > 0 and any(getattr(value, 'requires_grad', False) or bool(value != values[0]) for value in values):
            return True
    return False

def fold_k31_into_births(state, layer, mean_pre, variance_pre, row_keep, group_first, group_second, suffix_batch, suffix_squared, repeated_first, repeated_squared, repeated_suffix, star_vertex_batch, vertex_22_batch, vertex_31_batch, vertex_4o_batch, is_final, group_indices=None, fold_rows=None):
    runtime = state.runtime
    (block_group_first, block_group_second) = (group_first, group_second)
    k31_beta = float(PARAMS['exact_k31_betas'][min(max(layer, 1), 14)])
    if not is_final and k31_beta != 0.0 and (state.source_k4_family_vertices is not None) and (state.k4_family_ratios[0] != 0.0) and (state.k4_family_ratios[1] != 0.0) and (float(PARAMS['k4_d40o_scale']) != 0.0):
        (ratio_22c, ratio_31c, ratio_22g) = state.k4_family_ratios
        k31_scale = -k31_beta * mean_pre / fnp.maximum(variance_pre, VAR_FLOOR)
        if row_keep is not None:
            k31_scale = k31_scale[row_keep]
        if int(suffix_batch.shape[1]) < int(k31_scale.shape[0]):
            k31_scale = k31_scale[:int(suffix_batch.shape[1])]
        if k4_birth_normalization_needed(runtime):
            indices = range(len(state.source_k4_birth_ratios)) if group_indices is None else group_indices
            birth = [state.source_k4_birth_ratios[index] for index in indices]
            birth_g = fnp.asarray([ratios[2] for ratios in birth], dtype=star_vertex_batch.dtype)[:, None]
            birth_inv22 = fnp.asarray([1.0 / ratios[0] for ratios in birth], dtype=vertex_22_batch.dtype)[:, None]
            birth_inv31 = fnp.asarray([1.0 / ratios[1] for ratios in birth], dtype=vertex_31_batch.dtype)[:, None]
            k31_c22 = (vertex_22_batch - birth_g * star_vertex_batch) * birth_inv22
            k31_c31 = vertex_31_batch * birth_inv31
        else:
            k31_c22 = (vertex_22_batch - ratio_22g * star_vertex_batch) * (1.0 / ratio_22c)
            k31_c31 = vertex_31_batch * (1.0 / ratio_31c)
        # The covariance/history contractions read only their selected pair
        # rows. Far-row moments retain the original, unfolded blocks.
        # Restrict contiguous windows only: scattered gathers and inverse
        # restoration cost more than the arithmetic saved on a wide window.
        if fold_rows is not None and 10 * (fold_rows[1] - fold_rows[0]) >= 9 * int(group_first.shape[1]):
            fold_rows = None
        selection = slice(None) if fold_rows is None else slice(*fold_rows)
        t = suffix_batch[:, selection, :]
        t2 = suffix_squared[:, selection, :]
        q = repeated_first[:, selection, :]
        q2 = repeated_squared[:, selection, :]
        tq = repeated_suffix[:, selection, :]
        scale = k31_scale[selection][None, :, None]

        # Reuse the three quadratic terms in both cubic derivatives:
        # F = T * (c31*T^2 + 3*(c22*T*Q + star*Q^2))
        # S = T * (c4*T^2) + Q * (3*(c31*T^2+c22*T*Q)+star*Q^2).
        # No rank truncation or vertex changes are involved.
        c31_t2 = fnp.einsum('gij,gj->gij', t2, k31_c31, optimize=False)
        c22_tq = fnp.einsum('gij,gj->gij', tq, k31_c22, optimize=False)
        star_q2 = fnp.einsum('gij,gj->gij', q2, star_vertex_batch, optimize=False)
        c4_t2 = fnp.einsum('gij,gj->gij', t2, vertex_4o_batch, optimize=False)
        scaled_t = scale * t
        scaled_q = scale * q
        first_delta = scaled_t * (c31_t2 + 3.0 * (c22_tq + star_q2))
        second_delta = scaled_t * c4_t2 + scaled_q * (3.0 * (c31_t2 + c22_tq) + star_q2)
        if isinstance(fold_rows, tuple):
            first, last = fold_rows
            block_group_first = fnp.concatenate([group_first[:, :first, :], group_first[:, first:last, :] + first_delta, group_first[:, last:, :]], axis=1)
            block_group_second = fnp.concatenate([group_second[:, :first, :], group_second[:, first:last, :] + second_delta, group_second[:, last:, :]], axis=1)
        else:
            block_group_first = group_first + first_delta
            block_group_second = group_second + second_delta
    return (block_group_first, block_group_second)

def propagate_cumulants4(state, frame):
    layer = frame.layer
    weight = frame.weight
    core = state.runtime.coefficients['input_core'] if layer == 0 else state.k4_core
    frame.radial_diagonal = None
    if frame.is_final:
        if layer == 0 or state.identity_share == 0.0:
            frame.radial_diagonal = frame.variance_pre
        else:
            metric_diagonal = fnp.sum(frame.weight_squared, axis=0) if frame.weight_squared is not None else fnp.sum(weight * weight, axis=0)
            frame.radial_diagonal = state.identity_gain * metric_diagonal + state.cov_gain * frame.variance_pre
        frame.k4_diagonal = fnp.einsum(',i,i->i', core, frame.radial_diagonal, frame.radial_diagonal, optimize=False)
        frame.radial_k4 = frame.k4_diagonal
        frame.k4_22 = None
    else:
        if layer == 0 or state.identity_share == 0.0:
            radial = frame.covariance_pre
            frame.radial_diagonal = frame.variance_pre
        else:
            metric_diagonal = fnp.sum(frame.weight_squared, axis=0) if frame.weight_squared is not None else fnp.sum(weight * weight, axis=0)
            frame.radial_diagonal = state.identity_gain * metric_diagonal + state.cov_gain * frame.variance_pre
            radial = frame.covariance_pre
        frame.k4_diagonal = fnp.einsum(',i,i->i', core, frame.radial_diagonal, frame.radial_diagonal, optimize=False)
        frame.radial_k4 = frame.k4_diagonal
        frame.k4_22 = state.k4_core / 3.0 * (fnp.outer(frame.radial_diagonal, frame.radial_diagonal) + fnp.einsum('z,ij,ij->ij', state.runtime.coefficients['k4_off_twice'], radial, radial, optimize=False))
        frame.k4_22 = zero_diagonal_owned(frame.k4_22)
    if frame.star_channel is not None:
        frame.star_channel = frame.star_channel - fnp.mean(frame.star_channel)
        channel_total = state.k4_star_scale * frame.star_channel
        frame.k4_diagonal = frame.k4_diagonal + channel_total
        if state.k4_22_dress != 0.0 and frame.k4_22 is not None and (frame.radial_diagonal is not None):
            frame.k4_22 = dress_k4_pair(frame.k4_22, channel_total, frame.radial_diagonal, state.k4_22_dress)

def store_k4_vertices(state, frame):
    state.source_k4_birth_ratios.append(state.k4_family_ratios)
    if state.source_k4_star_vertices is not None:
        state.source_k4_star_vertices.append(frame.w31 if frame.alive is None else frame.w31[frame.alive])
    if state.source_k4_family_vertices is not None:
        (ratio_22c, ratio_31c, ratio_22g) = state.k4_family_ratios
        own40 = float(PARAMS['k4_d40o_scale']) != 0.0
        consistent_c31 = ratio_31c != 0.0
        if own40 or consistent_c31:
            (p2, p3, p4) = frame.power_rows[1:]
            p1_squared = frame.p1 * frame.p1 if frame.ordinary_shared is None else frame.ordinary_shared[0]
            c4_low = p4 - 4.0 * frame.p1 * p3 - 3.0 * p2 * p2 if frame.ordinary_shared is None else frame.ordinary_shared[2]
            c4 = c4_low + fnp.einsum('z,i,i->i', state.runtime.coefficients['twelve'], p1_squared, p2, optimize=False) - fnp.einsum('i,i,z->i', p1_squared, p1_squared, state.runtime.coefficients['six'], optimize=False)
        if ratio_31c != 0.0:
            (p2_c31, p3_c31) = (p2, p3)
            _pb_15 = frame.p1
            _pb_16 = p2_c31 - p1_squared if frame.ordinary_shared is None else frame.ordinary_shared[1]
            _pb_20 = fnp.einsum
            _pb_21 = state.runtime
            _pb_30 = frame.mean_pre
            _pb_34 = frame.variance_floor
            _pb_22 = _pb_21.coefficients
            _pb_23 = _pb_22['two']
            _pb_24 = _pb_20('z,i,i->i', _pb_23, _pb_15, p1_squared, optimize=False)
            _pb_17 = 3.0 * _pb_15
            _pb_18 = _pb_17 * p2_c31
            _pb_19 = p3_c31 - _pb_18
            _pb_25 = _pb_19 + _pb_24
            _pb_26 = _pb_17 * _pb_16
            _pb_27 = _pb_25 - _pb_26
            _pb_29 = fnp.polyval([_pb_15, _pb_27], p1_squared)
            _pb_31 = _pb_15 - _pb_30
            _pb_33 = fnp.polyval([_pb_29, c4], _pb_31)
            _pb_35 = _pb_33 / _pb_34
            m_c31 = _pb_15
            m2_c31 = p1_squared
            var_post_c31 = _pb_16
            three_m_c31 = _pb_17
            k3_post_c31 = _pb_25
            k4_post_c31 = c4
            tail_c31 = _pb_29
            c31 = _pb_35
        if ratio_22c != 0.0:
            c22 = 2.0 * (frame.w11 * (1.0 - frame.w11) - frame.w01 * frame.w21)
        vertex_22 = ratio_22c * c22 + ratio_22g * frame.w31 if ratio_22c != 0.0 and ratio_22g != 0.0 else ratio_22c * c22 if ratio_22c != 0.0 else ratio_22g * frame.w31 if ratio_22g != 0.0 else None
        vertex_31 = ratio_31c * c31 if ratio_31c != 0.0 else None
        vertex_4 = c4 if own40 else None
        state.source_k4_family_vertices.append(tuple((None if vertex is None else vertex if frame.alive is None else vertex[frame.alive] for vertex in (vertex_22, vertex_31, vertex_4))))

def update_k4_host(state, frame):
    (k4_diagonal_sum, k4_22_sum) = frame.k4_slices
    sigma_trace = fnp.sum(frame.k2_diagonal)
    sigma_frobenius = fnp.einsum('i,i->', frame.k2_diagonal, frame.k2_diagonal, optimize='optimal') + frame.offdiag_frobenius
    trace_squared = sigma_trace * sigma_trace
    radial_frobenius = state.identity_frobenius_share * trace_squared / float(state.width) + state.cov_frobenius_share * sigma_frobenius
    denominator = trace_squared + 2.0 * radial_frobenius
    state.k4_core = 3.0 * (k4_diagonal_sum + k4_22_sum) / denominator
    state.identity_gain = state.identity_share * sigma_trace / float(state.width)
    state.cov_gain = state.cov_weight
