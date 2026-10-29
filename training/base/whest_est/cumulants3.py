"""Third-cumulant leg transport, source births and shared-basis joins.
Math: source diagonal sums T*(Q**2*d+T**2*f+R*T)+6*u*T**2*Q; the 21 slice
combines source products and rank-two remainder. Inputs: source banks and frame.
Outputs: diagonal/21 cumulants, features and next factors. Knobs: vertex/carrier
gains, leg/final scales, band calibration and row/column/transport tiers."""
from __future__ import annotations
from .array_ops import prefix_view, column_rows, elementwise_stack, far_mean_rows, feature_sum, pad_axis, pair_rows, q_blocks, q_columns, q_front, zero_diagonal_owned
from .config import PARAMS
from .cumulants4 import fold_k31_into_births, reduce_k4_stars
from .history import birth_chain, consume_births
from .linalg import _fast_matmul, _matmul_operand
from .selection import restore_terminal_leg_rows, active_rows_matmul, count_alive, fill_active_rows, pad_product_rows, rank_at, select_transport, tail_count, terminal_fill, terminal_fill_setup, zero_inactive_rows
from .state import LEG_MEAN_FLOOR, ONE_THIRD, VAR_FLOOR, W01, W31, W41, W51, flops, fnp

def split_birth_blocks(block, indices, source_parts):
    result = [None for _ in indices]
    widths = {}
    for (offset, index) in enumerate(indices):
        widths.setdefault(int(source_parts[index].shape[1]), []).append(offset)
    for (columns, offsets) in widths.items():
        if columns == int(block.shape[2]):
            parts = fnp.unstack(block, axis=0)
            for offset in offsets:
                result[offset] = parts[offset]
        elif len(offsets) > 2:
            parts = fnp.unstack(block[:, :, :columns], axis=0)
            for offset in offsets:
                result[offset] = parts[offset]
        else:
            for offset in offsets:
                result[offset] = block[offset, :, :columns]
    return result

def terminal_birth_inputs(block, indices, rows=None):
    """Select live H12 sources and, for leg products, its already ordered rows."""
    if indices == list(range(indices[0], indices[-1] + 1)):
        sources = slice(indices[0], indices[-1] + 1)
        return block[sources] if rows is None else block[sources, rows, :]
    selected = block[indices]
    return selected if rows is None else selected[:, rows, :]

def diagonal_q_front(runtime, part, qoff, width):
    """Remember the paid wide padding for later birth-domain prefix reads."""
    result = q_front(part, qoff, width)
    if qoff > 0:
        runtime.readout.setdefault('wide_q_front', {})[id(part)] = (part, qoff, width, result)
    return result

def newborn_path(runtime, covariance, k3, w11, w21, ratio):
    inner = covariance + fnp.einsum('z,j,ji->ij', runtime.coefficients['heavy'], ratio, k3, optimize=False)
    return fnp.einsum('i,ij->ij', w11, inner, optimize=False) + fnp.einsum('z,i,ij->ij', runtime.coefficients['heavy'], w21, k3, optimize=False)

def path_tree(runtime, covariance, ratio, w21, layer):
    return fnp.einsum('z,j,i,ij,ij->ij', runtime.coefficients['t3'][layer], ratio, w21, covariance, covariance, optimize=False)

def carried_gain(runtime, w11, k3, k4, w41, w51, w71):
    return w11 + fnp.einsum('z,i,i->i', runtime.coefficients['gain3'], k3, w41, optimize=False) + fnp.einsum('z,i,i->i', runtime.coefficients['gain4'], k4, w51, optimize=False) + fnp.einsum('i,i,z,i->i', k3, k3, runtime.coefficients['gain33'], w71, optimize=False)

def weighted_leg_join(transport, path_factor, rank):
    columns = int(transport.shape[1])
    row_energy = fnp.sum(path_factor * path_factor, axis=1)
    weight = fnp.sqrt(1.0 + row_energy / (fnp.mean(row_energy) + 1e-30))
    work = fnp.einsum('ij,j->ij', transport, weight, optimize=False)
    (basis, values, right_transpose) = fnp.linalg.svd(work, full_matrices=False, k=rank - 1)
    projected = fnp.einsum('ji,j->ij', right_transpose, values, optimize=False)
    projected = projected / weight[:, None]
    ones = fnp.ones((columns, 1), dtype=fnp.float32)
    sums = _fast_matmul(transport, ones)
    coords = _fast_matmul(projected.T, ones)
    dropped = sums - _fast_matmul(basis, coords)
    direction = dropped / (fnp.sqrt(fnp.sum(dropped * dropped)) + LEG_MEAN_FLOOR)
    basis = fnp.concatenate([basis, direction], axis=1)
    projected = fnp.concatenate([projected, _fast_matmul(transport.T, direction)], axis=1)
    path_right = _fast_matmul(path_factor.T, projected)
    return (basis, projected, path_right)

def fixed_projection(runtime, width: int, rank: int) -> fnp.ndarray:
    if width != 1024 or rank != 2 or runtime.projection is None:
        raise ValueError('M1 requires its fixed production-shape projection asset')
    return runtime.projection

def remainder_factors(remainder: fnp.ndarray, projection: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    (basis, triangular) = fnp.linalg.qr(remainder @ projection, mode='reduced')
    remainder_t = remainder.T
    for _ in range(4):
        (adjoint, triangular) = fnp.linalg.qr(remainder_t @ basis, mode='reduced')
        (basis, triangular) = fnp.linalg.qr(remainder @ adjoint, mode='reduced')
    right = remainder_t @ basis
    return (basis, right)

def rebase_shared_legs(lefts, rights, paths, candidates, basis, ids, drop, lost_coordinates_t):
    if len(candidates) == 1:
        i = candidates[0]
        rights[i] = pad_axis(rights[i][:, ids], 0, 1, 1) + rights[i][:, drop:drop + 1] @ lost_coordinates_t
        paths[i] = pad_axis(paths[i][:, ids], 0, 1, 1) + paths[i][:, drop:drop + 1] @ lost_coordinates_t
        lefts[i] = basis
        return
    parts = [part for i in candidates for part in (rights[i], paths[i])]
    packed = fnp.concatenate(parts, axis=0)
    rebased = pad_axis(packed[:, ids], 0, 1, 1) + packed[:, drop:drop + 1] @ lost_coordinates_t
    boundaries = []
    rows = 0
    for part in parts[:-1]:
        rows += int(part.shape[0])
        boundaries.append(rows)
    pieces = fnp.split(rebased, boundaries, axis=0)
    for offset, i in enumerate(candidates):
        rights[i] = pieces[2 * offset]
        paths[i] = pieces[2 * offset + 1]
        lefts[i] = basis

def join_legs(runtime, transport, path_factor, rank, lefts, rights, paths, source_index, layer):
    runtime.join_reused[0] = False
    fused = runtime.fused_join.pop(source_index, None)

    def parent():
        full = transport if fused is None else _fast_matmul(fused[0], fused[1])
        return weighted_leg_join(full, path_factor, rank)
    candidates = [i for i in range(source_index) if lefts[i] is not None]
    if not candidates or not 8 <= layer <= 12:
        return parent()
    ref = lefts[candidates[0]]
    if int(ref.shape[1]) != int(rank) or any((lefts[i] is not ref for i in candidates)):
        return parent()
    k = int(ref.shape[1])
    drop = max(0, k - 8)
    ids = fnp.asarray([j for j in range(k) if j != drop], dtype=fnp.int64)
    kept = ref[:, ids]
    kept_t = kept.T
    lost = ref[:, drop:drop + 1]
    solve_right = None
    if fused is not None:
        (a, x) = fused
        gram = fnp.einsum('ji,jk->ik', kept, kept)
        rhs = _fast_matmul(kept_t, a)
        if int(x.shape[0]) < int(x.shape[1]):
            # solve(G, H @ X) = solve(G, H) @ X. Solve only the
            # transport-width columns when the source was born wider.
            solve_right = x
        else:
            rhs = _fast_matmul(rhs, x)
        mean = _fast_matmul(a, fnp.mean(x, axis=1)[:, None])
    else:
        gram = fnp.einsum('ji,jk->ik', kept, kept)
        rhs = kept_t @ transport
        mean = fnp.mean(transport, axis=1)[:, None]
    ridge = 1e-06 * fnp.mean(fnp.diag(gram))
    work = gram + ridge * fnp.eye(k - 1, dtype=fnp.float32)
    _cf2_rhs = fnp.concatenate([rhs, kept_t @ lost], axis=1)
    _cf2_sol = fnp.linalg.solve(work, _cf2_rhs)
    coordinates = _cf2_sol[:, :int(rhs.shape[1])]
    _cf2_lost_coordinates = _cf2_sol[:, int(rhs.shape[1]):]
    if solve_right is not None:
        coordinates = _fast_matmul(coordinates, solve_right)
    residual = mean - kept @ fnp.mean(coordinates, axis=1)[:, None]
    direction = residual / (fnp.sqrt(fnp.sum(residual * residual)) + LEG_MEAN_FLOOR)
    basis = fnp.concatenate([kept, direction], axis=1)
    if fused is None:
        direction_coordinates = transport.T @ direction
    else:
        direction_coordinates = _fast_matmul(x.T, _fast_matmul(a.T, direction))
    lost_kept = _cf2_lost_coordinates
    lost_direction = direction.T @ lost
    overlap = kept_t @ direction
    coordinates_t = coordinates.T
    direction_coordinates = direction_coordinates - coordinates_t @ overlap
    lost_direction = lost_direction - overlap.T @ lost_kept
    projected = fnp.concatenate([coordinates_t, direction_coordinates], axis=1)
    lost_coordinates = fnp.concatenate([lost_kept, lost_direction], axis=0)
    lost_coordinates_t = lost_coordinates.T
    rebase_shared_legs(lefts, rights, paths, candidates, basis, ids, drop, lost_coordinates_t)
    path_right = _fast_matmul(path_factor.T, projected)
    runtime.join_reused[0] = True
    return (basis, projected, path_right)

def fused_join_index(runtime, layer, dense, lefts, q_indices, tiers, width, depth):
    runtime.fused_join.clear()
    if width != 1024 or depth != 16 or (not 8 <= layer <= 12):
        return None
    candidates = [i for i in range(layer - 1) if dense[i] is not None and rank_at(tiers, layer - i, width) is not None and (i not in q_indices)]
    owners = [i for i in range(layer - 1) if lefts[i] is not None]
    if len(candidates) != 1 or not owners:
        return None
    ref = lefts[owners[0]]
    if any((lefts[i] is not ref for i in owners)):
        return None
    i = candidates[0]
    if int(ref.shape[1]) != int(rank_at(tiers, layer - i, width)):
        return None
    return i

def capture_fused_join(runtime, index, old, a, count, ids, gain=None):
    if index is None:
        return
    x = old if count is None and ids is None else prefix_view(old, count) if count is not None else old[ids, :]
    if gain is not None:
        x = fnp.einsum('i,ij->ij', gain, x, optimize=False)
    runtime.fused_join[index] = (a, x)

def leg_features(t2, q2):
    return (fnp.einsum('gij->i', t2), fnp.einsum('gij->i', q2))

def finish_leg_features(acc, expand, mask, inverse, rows, width, term_inverse, term_rows):
    if not acc:
        return None
    cols = [feature_sum((x[j] for x in acc)) for j in (0, 1)]
    out = fnp.stack(cols, axis=1)
    if expand and mask is not None:
        out = fnp.einsum('i,ij->ij', mask.astype(out.dtype), out, optimize=False)
    if inverse is not None:
        if rows < width:
            out = pad_axis(out, 0, width - rows, 0)
        out = out[inverse]
    if not (term_inverse is None or expand):
        out = pad_axis(out, 0, width - term_rows, 0)[term_inverse]
    return out

class ProjectionWeights(flops.Module):

    def __init__(self):
        self.production = fnp.zeros((), dtype=fnp.float32)

    def load_state_dict(self, sd, *, strict=True):
        # Older source packages contain a retired 4 MiB signs matrix.
        sd = {key: value for key, value in sd.items() if key != 'signs'}
        super().load_state_dict(sd, strict=strict)

def load_projection(runtime, directory):
    runtime.projection = None
    if directory is None:
        return
    from pathlib import Path as _P
    path = _P(str(directory)) / 'hadamard.npz'
    if not path.exists():
        raise ValueError('AM projection requires hadamard.npz')
    w = ProjectionWeights.from_file(str(path))
    if tuple(w.production.shape) != (1024, 2):
        raise ValueError('invalid precomputed Hadamard data')
    runtime.projection = w.production

def transport_cumulants3(state, frame):
    runtime = state.runtime
    layer = frame.layer
    weight = frame.weight
    if layer == 0:
        frame.pre_transport = None
        frame.k3_diagonal = state.zeros_vector
        frame.k3_21 = fnp.zeros((state.width, state.width), dtype=fnp.float32)
    else:
        terminal_leg_rows = terminal_leg_inverse = None
        if frame.expand_terminal_births and frame.terminal_inverse_order is not None:
            # Expanded terminal births require the union of the terminal
            # diagonal rows and the global H12 readout rows, not all neurons.
            # Every other terminal leg row is masked before a cross-row use.
            birth_inverse = fnp.argsort(frame.birth_order, stable=True)
            required_rows = fnp.logical_or(frame.terminal_inverse_order < state.terminal_kink_rows, birth_inverse < 128)
            union_count = (int(fnp.sum(required_rows)) + 63) // 64 * 64
            if union_count < int(state.width):
                union_order = fnp.argsort(fnp.logical_not(required_rows), stable=True)
                terminal_leg_rows = union_order[:union_count]
                terminal_leg_inverse = fnp.argsort(union_order, stable=True)
        alive_prev = state.source_alive[layer - 1]
        frame.transport_alive = state.source_transport_alive[layer - 1]
        newborn_alive = alive_prev
        _bugc_grows = int(state.width) if frame.transport_alive is None else int(frame.transport_alive.shape[0])
        if newborn_alive is None:
            newborn_weight = weight
        else:
            newborn_weight = weight[newborn_alive, :]
        if frame.terminal_selected is not None and (not frame.expand_terminal_births):
            newborn_weight = newborn_weight[:, frame.terminal_selected]
        if frame.row_keep is not None:
            newborn_weight = newborn_weight[:, frame.row_keep]
        if terminal_leg_rows is not None:
            newborn_weight = newborn_weight[:, terminal_leg_rows]
        newborn_weight_t = newborn_weight.T
        newborn_leg = zero_inactive_rows(newborn_weight_t, frame.active_rows)
        newborn_leg = restore_terminal_leg_rows(newborn_leg, terminal_leg_inverse, state.width, 0)
        if state.row_inverse_prev is None:
            alive_prev = frame.transport_alive
            birth_rows = newborn_alive
            transport_rows = None
            birth_count = None
        else:
            alive_prev = None
            birth_rows = None
            transport_rows = None if frame.transport_alive is None else int(frame.transport_alive.shape[0])
            birth_count = None if newborn_alive is None else int(newborn_alive.shape[0])
        early_product = getattr(frame, 'early_leg_product', None)
        frame.early_leg_product = None
        newborn_q = None
        if layer > 1:
            dense_indices = [index for index in range(layer - 1) if state.leg_dense[index] is not None]
            q_indices = [index for index in dense_indices if layer >= state.mlp.depth - 3 or rank_at(state.leg_tiers, layer - index, state.width) is None]
            active_ranks = sorted({int(state.leg_left[index].shape[1]) for index in range(layer - 1) if state.leg_left[index] is not None}, reverse=True)
            _r_fuse = fused_join_index(runtime, layer, state.leg_dense, state.leg_left, q_indices, state.leg_tiers, int(state.width), int(state.mlp.depth))
            descriptors = [('t', index, state.leg_dense[index]) for index in dense_indices if index != _r_fuse]
            descriptors += [('q', index, state.leg_q_dense[index]) for index in q_indices]
            for active_rank in active_ranks:
                descriptors += [('l', index, state.leg_left[index]) for index in range(layer - 1) if state.leg_left[index] is not None and int(state.leg_left[index].shape[1]) == active_rank]
            _v144_basis_owners = {}
            _v144_unique = []
            for (kind, index, part) in descriptors:
                if kind != 'l':
                    _v144_unique.append((kind, index, part))
                elif id(part) in _v144_basis_owners:
                    _v144_basis_owners[id(part)].append(index)
                else:
                    owner = [index]
                    _v144_basis_owners[id(part)] = owner
                    _v144_unique.append((kind, owner, part))
            descriptors = _v144_unique
            share_birth = int(newborn_weight.shape[0]) == _bugc_grows
            if early_product is not None:
                packed_all = early_product
            elif share_birth:
                if birth_count is not None:
                    parts = [prefix_view(part, birth_count) for (kind, index, part) in descriptors]
                else:
                    parts = [part if birth_rows is None else part[birth_rows, :] for (kind, index, part) in descriptors]
                packed_old = fnp.concatenate(parts, axis=1)
                sorted_gain = state.carried_gain if newborn_alive is None else state.carried_gain[newborn_alive]
                packed_rhs = fnp.concatenate([fnp.einsum('i,ij->ij', sorted_gain, packed_old, optimize=False), state.source_path_factors[layer - 1]], axis=1)
                if _r_fuse is not None:
                    capture_fused_join(runtime, _r_fuse, state.leg_dense[_r_fuse], newborn_leg, birth_count, birth_rows, sorted_gain)
                packed_all = active_rows_matmul(newborn_weight_t, packed_rhs, frame.active_rows)
            else:
                (gained_weight, gained_gain) = select_transport(weight, state.carried_gain, frame.transport_alive, frame.terminal_selected, frame.expand_terminal_births, frame.row_keep)
                transport_weight = fnp.einsum('i,ij->ij', gained_gain, gained_weight, optimize=False).T
                if terminal_leg_rows is not None:
                    transport_weight = transport_weight[terminal_leg_rows, :]
                if transport_rows is not None:
                    parts = [prefix_view(part, transport_rows) for (kind, index, part) in descriptors]
                else:
                    parts = [part if alive_prev is None else part[alive_prev, :] for (kind, index, part) in descriptors]
                if _r_fuse is not None:
                    capture_fused_join(runtime, _r_fuse, state.leg_dense[_r_fuse], zero_inactive_rows(transport_weight, frame.active_rows), transport_rows, alive_prev)
                packed_all = active_rows_matmul(transport_weight, fnp.concatenate(parts, axis=1), frame.active_rows)
            packed_all = restore_terminal_leg_rows(packed_all, terminal_leg_inverse, state.width, 0)
            offset = 0
            split_points = []
            for (kind, index, part) in descriptors:
                offset += int(part.shape[1])
                if offset < int(packed_all.shape[1]):
                    split_points.append(offset)
            descriptor_parts = fnp.split(packed_all, split_points, axis=1)
            if share_birth:
                newborn_q = descriptor_parts[-1]
            offset = 0
            for (descriptor_offset, (kind, index, part)) in enumerate(descriptors):
                columns = int(part.shape[1])
                value = descriptor_parts[descriptor_offset]
                if kind == 't':
                    state.leg_dense[index] = value
                elif kind == 'q':
                    state.leg_q_dense[index] = value
                else:
                    for _v144_owner in index:
                        state.leg_left[_v144_owner] = value
                offset += columns
        state.leg_dense.append(newborn_leg)
        state.leg_q_dense.append(newborn_q if layer > 1 else None)
        state.leg_left.append(None)
        state.leg_right.append(None)
        state.leg_path_right.append(None)
        for source_index in range(layer):
            age = layer - source_index
            target_rank = rank_at(state.leg_tiers, age, state.width)
            if target_rank is None:
                continue
            current_left = state.leg_left[source_index]
            if current_left is None and layer >= state.mlp.depth - 3:
                continue
            if current_left is None:
                runtime.join_reused[0] = False
                (left, right, path_right) = join_legs(runtime, state.leg_dense[source_index], state.source_path_factors[source_index], target_rank, state.leg_left, state.leg_right, state.leg_path_right, source_index, layer)
                if state.leg_level_gain != 0.0:
                    tail_energy = PARAMS['leg_tail_energy'].get(str(age) + ':' + str(target_rank))
                    if tail_energy is None:
                        raise ValueError('no universal tail-energy entry for active leg tier')
                    if not runtime.join_reused[0]:
                        left = fnp.sqrt(fnp.asarray(1.0 + state.leg_level_gain * tail_energy / 2.0, dtype=fnp.float32)) * left
                state.leg_dense[source_index] = None
                state.leg_q_dense[source_index] = None
                state.leg_left[source_index] = left
                state.leg_right[source_index] = right
                state.leg_path_right[source_index] = path_right
        frame.suffix_parts = [None for _ in range(layer)]
        suffix_full_parts = [None for _ in range(layer)]
        frame.repeated_parts = [None for _ in range(layer)]
        dense_indices = [index for index in range(layer) if state.leg_dense[index] is not None]
        dense_widths = sorted({int(state.leg_dense[index].shape[1]) for index in dense_indices}, reverse=True)
        for dense_width in dense_widths:
            width_indices = [index for index in dense_indices if int(state.leg_dense[index].shape[1]) == dense_width]
            missing_q = [index for index in width_indices if state.leg_q_dense[index] is None]
            if missing_q:
                fresh_suffix = fnp.stack([state.leg_dense[index] for index in missing_q], axis=0)
                fresh_path = fnp.stack([state.source_path_factors[index] for index in missing_q], axis=0)
                # Dense legs already have an exact zero tail from active-row
                # transport. Do not multiply that padding by the birth path.
                fresh_rows_count = int(fresh_suffix.shape[1])
                if terminal_leg_rows is not None:
                    fresh_q = _fast_matmul(fresh_suffix[:, terminal_leg_rows, :], fresh_path)
                    fresh_q = restore_terminal_leg_rows(fresh_q, terminal_leg_inverse, state.width, 1)
                elif frame.active_rows is not None and frame.active_rows[0] < fresh_rows_count:
                    fresh_q = _fast_matmul(fresh_suffix[:, :frame.active_rows[0], :], fresh_path)
                    fresh_q = pad_product_rows(fresh_q, 0, fresh_rows_count)
                else:
                    fresh_q = _fast_matmul(fresh_suffix, fresh_path)
                fresh_rows = fnp.unstack(fresh_q, axis=0) if len(missing_q) > 1 else None
                for (q_offset, index) in enumerate(missing_q):
                    state.leg_q_dense[index] = fresh_rows[q_offset] if fresh_rows is not None else fresh_q[q_offset]
            for index in width_indices:
                suffix_full_parts[index] = state.leg_dense[index]
                frame.suffix_parts[index] = state.leg_dense[index]
                frame.repeated_parts[index] = state.leg_q_dense[index]
        factored_keys = sorted({(int(state.leg_left[index].shape[1]), int(state.leg_right[index].shape[0]), state.q_offsets[index]) for index in range(layer) if state.leg_left[index] is not None}, reverse=True)
        factored_groups = []
        for (active_rank, active_width, active_qoff) in factored_keys:
            rank_indices = [index for index in range(layer) if state.leg_left[index] is not None and int(state.leg_left[index].shape[1]) == active_rank and (int(state.leg_right[index].shape[0]) == active_width) and (state.q_offsets[index] == active_qoff)]
            factored_groups.append(rank_indices)
            repeated_width = active_width - active_qoff
            left_batch = _matmul_operand([state.leg_left[index] for index in rank_indices])
            _fe52_lb_rows = int(left_batch.shape[1])
            _fe52_cut = frame.active_rows is not None and frame.active_rows[0] < _fe52_lb_rows
            if _fe52_cut:
                left_batch = left_batch[:, 0:min(frame.active_rows[0], _fe52_lb_rows), :]
            if terminal_leg_rows is not None:
                left_batch = left_batch[:, terminal_leg_rows, :]
            right_pair_batch = fnp.stack([fnp.concatenate([state.leg_path_right[index], state.leg_right[index]], axis=0) for index in rank_indices], axis=0).transpose(0, 2, 1)
            materialized_pair = _fast_matmul(left_batch, right_pair_batch)
            materialized_pair = restore_terminal_leg_rows(materialized_pair, terminal_leg_inverse, state.width, 1)
            if _fe52_cut:
                materialized_pair = pad_product_rows(materialized_pair, 0, _fe52_lb_rows)
            if len(rank_indices) > 2:
                (repeated_batch, suffix_batch) = fnp.split(materialized_pair, [repeated_width], axis=2)
                repeated_rows = fnp.unstack(repeated_batch, axis=0)
                suffix_rows = fnp.unstack(suffix_batch, axis=0)
            else:
                repeated_rows = suffix_rows = None
            for (offset, index) in enumerate(rank_indices):
                frame.repeated_parts[index] = repeated_rows[offset] if repeated_rows is not None else materialized_pair[offset, :, :repeated_width]
                suffix_full_parts[index] = suffix_rows[offset] if suffix_rows is not None else materialized_pair[offset, :, repeated_width:]
                frame.suffix_parts[index] = suffix_full_parts[index]
        frame.groups = []
        _f31_acc = []
        moment_groups = []
        selected_k3_diagonal = None
        frame.block_first = [None for _ in range(layer)]
        block_second = [None for _ in range(layer)]
        if frame.terminal_births:
            _h12_order = frame.birth_order if frame.expand_terminal_births else fnp.argsort(-fnp.abs(frame.wick[W41][frame.terminal_selected]), stable=True)
            _h12_ids = _h12_order[:128]
            _h12_inverse = fnp.argsort(_h12_order, stable=True)
        column_width = full_width = state.width
        group_indices = list(range(layer))
        suffix_batch = elementwise_stack([frame.suffix_parts[index] for index in group_indices], column_width, -1)
        suffix_full_batch = suffix_batch
        repeated_first = elementwise_stack([diagonal_q_front(runtime, p, o, column_width) for (p, o) in zip([frame.repeated_parts[index] for index in group_indices], [state.q_offsets[index] for index in group_indices])], column_width, -1)
        # Active-row transport and materialization guarantee an exactly zero
        # tail. Preserve the complete legs for history while evaluating all
        # pointwise source polynomials only on their nonzero prefix.
        _poly_n = int(suffix_batch.shape[1])
        _poly_cut = (frame.active_rows is not None and frame.pair_window is not None
                     and int(frame.active_rows[0]) < _poly_n
                     and frame.pair_window[1] <= int(frame.active_rows[0])
                     and not frame.is_final and frame.star_channel is None
                     and frame.family_channel is None)
        _poly_tail = _poly_n - int(frame.active_rows[0]) if _poly_cut else 0
        _poly_full_t, _poly_full_q = suffix_batch, repeated_first
        if _poly_cut:
            suffix_batch = suffix_batch[:, :frame.active_rows[0], :]
            suffix_full_batch = suffix_batch
            repeated_first = repeated_first[:, :frame.active_rows[0], :]
        free_u_batch = elementwise_stack([state.source_free_u[index] for index in group_indices], column_width, 0, cache=state.source_padding)
        free_diagonal_batch = elementwise_stack([state.source_free_diagonals[index] for index in group_indices], column_width, 0, cache=state.source_padding)
        remainder_left_batch = elementwise_stack([state.source_remainder_left[index] for index in group_indices], full_width, 0, cache=state.source_padding)
        remainder_right_batch = elementwise_stack([state.source_remainder_right[index] for index in group_indices], column_width, 0, cache=state.source_padding)
        transported_left_batch = _fast_matmul(suffix_full_batch, remainder_left_batch)
        transported_remainder_batch = _fast_matmul(transported_left_batch, remainder_right_batch.transpose(0, 2, 1))
        singleton_diagonal_batch = elementwise_stack([state.source_singleton_diagonals[index] for index in group_indices], column_width, 0, cache=state.source_padding)
        # Every history/slice consumer needs the full squared suffix. Square
        # it once, then borrow its live prefix for pointwise contractions.
        _poly_full_t2 = _poly_full_t * _poly_full_t
        suffix_squared = _poly_full_t2[:, :frame.active_rows[0], :] if _poly_cut else _poly_full_t2
        repeated_squared = repeated_first * repeated_first
        _poly_full_tl = pad_axis(transported_left_batch, 0, _poly_tail, 1) if _poly_cut else transported_left_batch
        repeated_square_diagonal = fnp.einsum('...ij,...j->...ij', repeated_squared, singleton_diagonal_batch, optimize=False)
        suffix_square_diagonal = fnp.einsum('...ij,...j->...ij', suffix_squared, free_diagonal_batch, optimize=False)
        free_suffix_square = fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['six'], free_u_batch, suffix_squared, optimize=False)
        remainder_suffix = transported_remainder_batch * suffix_batch
        repeated_suffix = repeated_first * suffix_batch
        if not frame.is_final or frame.terminal_births:
            birth_group_indices = group_indices
            if frame.terminal_births:
                birth_group_indices = [i for i in dense_indices if i < layer - 1]
                birth_singleton = terminal_birth_inputs(singleton_diagonal_batch, birth_group_indices)
                birth_free_u = terminal_birth_inputs(free_u_batch, birth_group_indices)
                birth_repeated_suffix = terminal_birth_inputs(repeated_suffix, birth_group_indices, _h12_ids)
                birth_free_suffix_square = terminal_birth_inputs(free_suffix_square, birth_group_indices, _h12_ids)
                birth_repeated_square_diagonal = terminal_birth_inputs(repeated_square_diagonal, birth_group_indices, _h12_ids)
                birth_suffix_square_diagonal = terminal_birth_inputs(suffix_square_diagonal, birth_group_indices, _h12_ids)
                birth_remainder_suffix = terminal_birth_inputs(remainder_suffix, birth_group_indices, _h12_ids)
            else:
                (birth_singleton, birth_free_u) = (singleton_diagonal_batch, free_u_batch)
                (birth_repeated_suffix, birth_free_suffix_square) = (repeated_suffix, free_suffix_square)
                birth_repeated_square_diagonal = repeated_square_diagonal
                (birth_suffix_square_diagonal, birth_remainder_suffix) = (suffix_square_diagonal, remainder_suffix)
            group_first = fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['two'], birth_singleton, birth_repeated_suffix, optimize=False) + birth_free_suffix_square
            group_second = fnp.einsum('z,...j,...ij->...ij', runtime.coefficients['twelve'], birth_free_u, birth_repeated_suffix, optimize=False) + birth_repeated_square_diagonal + 3.0 * birth_suffix_square_diagonal + 2.0 * birth_remainder_suffix
            if frame.pair_selected is not None:
                if _poly_cut:
                    moment_groups.append((group_first, group_second, _poly_full_q, _poly_full_t, _poly_full_t2, remainder_right_batch, _poly_full_tl))
                else:
                    moment_groups.append((group_first, group_second, repeated_first, suffix_batch, suffix_squared, remainder_right_batch, transported_left_batch))
        diagonal_terms = suffix_batch * (repeated_square_diagonal + suffix_square_diagonal + remainder_suffix) + free_suffix_square * repeated_first
        (frame.star_channel, frame.family_channel, star_vertex_batch, vertex_22_batch, vertex_31_batch, vertex_4o_batch) = reduce_k4_stars(state, group_indices, frame.suffix_parts, suffix_squared, repeated_squared, repeated_suffix, frame.star_channel, frame.family_channel, column_width)
        if not frame.is_final or frame.terminal_births:
            (block_group_first, block_group_second) = fold_k31_into_births(state, layer, frame.mean_pre, frame.variance_pre, frame.row_keep, group_first, group_second, suffix_batch, suffix_squared, repeated_first, repeated_squared, repeated_suffix, star_vertex_batch, vertex_22_batch, vertex_31_batch, vertex_4o_batch, frame.is_final, group_indices=group_indices, fold_rows=frame.pair_window)
            # Only the certified pair window reads these folded blocks, so
            # the omitted zero tail never needs a source-by-vertex expansion.
            _fe45_first_parts = split_birth_blocks(block_group_first, birth_group_indices, frame.suffix_parts)
            _fe45_second_parts = split_birth_blocks(block_group_second, birth_group_indices, frame.suffix_parts)
            for (offset, index) in enumerate(birth_group_indices):
                frame.block_first[index] = _fe45_first_parts[offset]
                block_second[index] = _fe45_second_parts[offset]
        _av_diagonal_by_source = fnp.sum(diagonal_terms, axis=2)
        group_diagonal = fnp.sum(_av_diagonal_by_source, axis=0)
        _f31_acc.append(leg_features(suffix_squared, repeated_squared))
        if _poly_cut:
            group_diagonal = pad_axis(group_diagonal, 0, _poly_tail, 0)
            _f31_acc = [tuple(pad_axis(value, 0, _poly_tail, 0) for value in item) for item in _f31_acc]
            if frame.star_channel is not None:
                frame.star_channel = pad_axis(frame.star_channel, 0, _poly_tail, 0)
            if frame.family_channel is not None:
                frame.family_channel = pad_axis(frame.family_channel, 0, _poly_tail, 0)
        selected_k3_diagonal = group_diagonal if selected_k3_diagonal is None else selected_k3_diagonal + group_diagonal
        frame.groups.append((_poly_full_t2, remainder_right_batch, _poly_full_tl))
        if frame.active_rows is not None:
            (selected_k3_diagonal, frame.star_channel, frame.family_channel, _f31_acc) = fill_active_rows(frame.active_rows, selected_k3_diagonal, frame.star_channel, frame.family_channel, _f31_acc, frame.gaussian[1][frame.row_keep])
        if frame.expand_terminal_births and frame.terminal_inverse_order is not None:
            _h12_mask = frame.terminal_inverse_order < state.terminal_kink_rows
            selected_k3_diagonal = fnp.where(_h12_mask, selected_k3_diagonal, state.zeros_vector)
            if frame.star_channel is not None:
                frame.star_channel = fnp.where(_h12_mask, frame.star_channel, state.zeros_vector)
            if frame.family_channel is not None:
                frame.family_channel = fnp.where(_h12_mask, frame.family_channel, state.zeros_vector)
        mask_subset = frame.expand_terminal_births and frame.terminal_inverse_order is not None
        runtime.features['legs'] = finish_leg_features(_f31_acc, mask_subset, _h12_mask if mask_subset else None, frame.row_inverse, frame.row_space, state.width, frame.terminal_inverse_order, state.terminal_kink_rows)
        _f31_acc = None
        row_space_k3_diagonal = selected_k3_diagonal
        if frame.row_inverse is not None:
            if frame.row_space < state.width:
                selected_k3_diagonal = pad_axis(selected_k3_diagonal, 0, state.width - frame.row_space, 0)
            selected_k3_diagonal = selected_k3_diagonal[frame.row_inverse]
            if frame.star_channel is not None:
                if frame.row_space < state.width:
                    frame.star_channel = pad_axis(frame.star_channel, 0, state.width - frame.row_space, 0)
                frame.star_channel = frame.star_channel[frame.row_inverse]
            if frame.family_channel is not None:
                if frame.row_space < state.width:
                    frame.family_channel = pad_axis(frame.family_channel, 0, state.width - frame.row_space, 0)
                frame.family_channel = frame.family_channel[frame.row_inverse]
        if frame.terminal_inverse_order is None or frame.expand_terminal_births:
            frame.k3_diagonal = selected_k3_diagonal
        else:
            _fe52_tf = terminal_fill_setup(frame.terminal_order, state.terminal_kink_rows, frame.mean_pre, frame.gaussian[1]) if frame.is_final else None
            ordered_k3_diagonal = pad_axis(selected_k3_diagonal, 0, state.width - state.terminal_kink_rows, 0) if _fe52_tf is None else terminal_fill(selected_k3_diagonal, _fe52_tf, 3)
            frame.k3_diagonal = ordered_k3_diagonal[frame.terminal_inverse_order]
            if frame.star_channel is not None:
                frame.star_channel = (pad_axis(frame.star_channel, 0, state.width - state.terminal_kink_rows, 0) if _fe52_tf is None else terminal_fill(frame.star_channel, _fe52_tf, 4))[frame.terminal_inverse_order]
            if frame.family_channel is not None:
                frame.family_channel = (pad_axis(frame.family_channel, 0, state.width - state.terminal_kink_rows, 0) if _fe52_tf is None else terminal_fill(frame.family_channel, _fe52_tf, 4))[frame.terminal_inverse_order]
        if frame.is_final:
            frame.k3_21 = None
            if frame.terminal_births:
                (_h12_unused, _h12_far, _h12_hs) = birth_chain(runtime, q_blocks(frame.block_first, state.q_offsets), block_second, frame.suffix_parts, frame.repeated_parts, state.leg_left, state.leg_right, state.leg_path_right, state.source_remainder_right, state.source_remainder_left, state.birth_history, _h12_ids, None, None, int(frame.suffix_parts[0].shape[0]), layer, True, state.birth_self_c, state.birth_c2, frame.mean_pre, frame.variance_floor, blocks_selected=True)
                frame.birth_diagonal = consume_births(runtime, _h12_hs, frame.suffix_parts, frame.repeated_parts, state.birth_self_c, state.birth_c2, _h12_ids, _h12_inverse, None if frame.expand_terminal_births else frame.terminal_inverse_order, int(frame.suffix_parts[0].shape[0]), state.width)
        else:
            if frame.pair_window is not None:
                (pair_first, pair_last) = frame.pair_window
            runtime.remainder_products.clear()
            (suffix_squared, remainder_right_batch, transported_left_batch) = frame.groups[0]
            if frame.pair_window is not None:
                squared_right_input = suffix_squared[:, pair_first:pair_last, :]
            elif frame.pair_selected is None:
                squared_right_input = suffix_squared
            else:
                squared_right_input = suffix_squared[:, frame.pair_selected, :]
            runtime.remainder_products.append(_fast_matmul(squared_right_input, remainder_right_batch))
            runtime.remainder_rows = fnp.unstack(runtime.remainder_products[0], axis=0) if runtime.remainder_products else None
            if frame.births_active:
                (slice_sum, far_column, _h12_hs) = birth_chain(runtime, q_blocks(frame.block_first, state.q_offsets), block_second, frame.suffix_parts, frame.repeated_parts, state.leg_left, state.leg_right, state.leg_path_right, state.source_remainder_right, state.source_remainder_left, state.birth_history, frame.pair_window if frame.pair_window is not None else frame.pair_selected, frame.row_keep, frame.col_count, frame.row_space, layer, False, state.birth_self_c, state.birth_c2, frame.mean_pre, frame.variance_floor)
                frame.birth_diagonal = consume_births(runtime, _h12_hs, frame.suffix_parts, frame.repeated_parts, state.birth_self_c, state.birth_c2, frame.pair_window if frame.pair_window is not None else frame.pair_selected, frame.pair_inverse_order, frame.row_inverse, frame.row_space, state.width)
                if runtime.older_births:
                    _f31_hrow_keep = runtime.features.get('hrow')
                    runtime.birth_diagonal[layer] = consume_births(runtime, list(runtime.older_births), frame.suffix_parts, frame.repeated_parts, state.birth_self_c, state.birth_c2, frame.pair_window if frame.pair_window is not None else frame.pair_selected, frame.pair_inverse_order, frame.row_inverse, frame.row_space, state.width)
                    runtime.older_births.clear()
                    runtime.features['hrow30'] = runtime.features.get('hrow')
                    runtime.features['hrow30_layer'] = layer
                    runtime.features['hrow'] = _f31_hrow_keep
            if not frame.births_active:
                dense_singleton = fnp.concatenate([pair_rows(q_columns(frame.block_first[index], state.q_offsets[index]), frame.pair_selected, frame.pair_window) for index in dense_indices] + [pair_rows(block_second[index], frame.pair_selected, frame.pair_window) for index in dense_indices], axis=1)
                dense_repeated = fnp.concatenate([column_rows(frame.repeated_parts[index], frame.col_count) for index in dense_indices] + [column_rows(frame.suffix_parts[index], frame.col_count) for index in dense_indices], axis=1)
                slice_sum = _fast_matmul(dense_singleton, dense_repeated.T)
                far_column = None
                if frame.col_count is not None and frame.col_count < frame.row_space:
                    dense_far = fnp.concatenate([far_mean_rows(frame.repeated_parts[index], frame.col_count) for index in dense_indices] + [far_mean_rows(frame.suffix_parts[index], frame.col_count) for index in dense_indices], axis=0)
                    far_column = _fast_matmul(dense_singleton, dense_far[:, None])
                for rank_indices in factored_groups:
                    xy_batch = fnp.stack([fnp.concatenate([pair_rows(q_columns(frame.block_first[index], state.q_offsets[index]), frame.pair_selected, frame.pair_window), pair_rows(block_second[index], frame.pair_selected, frame.pair_window)], axis=1) for index in rank_indices], axis=0)
                    cb_batch = fnp.stack([fnp.concatenate([state.leg_path_right[index], state.leg_right[index]], axis=0) for index in rank_indices], axis=0)
                    thin_slice_left = _fast_matmul(xy_batch, cb_batch)
                    packed_thin_left = fnp.concatenate([thin_slice_left[offset] for offset in range(len(rank_indices))], axis=1)
                    packed_thin_right = fnp.concatenate([column_rows(state.leg_left[index], frame.col_count).T for index in rank_indices], axis=0)
                    slice_sum = slice_sum + _fast_matmul(packed_thin_left, packed_thin_right)
                    if far_column is not None:
                        thin_far = fnp.concatenate([far_mean_rows(state.leg_left[index], frame.col_count) for index in rank_indices], axis=0)
                        far_column = far_column + _fast_matmul(packed_thin_left, thin_far[:, None])
            remainder_slice = None
            for (suffix_squared, remainder_right_batch, transported_left_batch) in frame.groups:
                if runtime.remainder_products:
                    squared_right_batch = runtime.remainder_products.pop(0)
                else:
                    if frame.pair_window is not None:
                        squared_right_input = suffix_squared[:, pair_first:pair_last, :]
                    elif frame.pair_selected is None:
                        squared_right_input = suffix_squared
                    else:
                        squared_right_input = suffix_squared[:, frame.pair_selected, :]
                    squared_right_batch = _fast_matmul(squared_right_input, remainder_right_batch)
                group_slice = fnp.einsum('sir,sjr->ij', squared_right_batch, transported_left_batch if frame.col_count is None else prefix_view(transported_left_batch, frame.col_count, 1), optimize='optimal')
                remainder_slice = group_slice if remainder_slice is None else remainder_slice + group_slice
                if far_column is not None:
                    group_far = fnp.einsum('sir,sr->i', squared_right_batch, fnp.mean(transported_left_batch[:, frame.col_count:, :], axis=1), optimize='optimal')
                    far_column = far_column + group_far[:, None]
            selected_k3_21 = ONE_THIRD * (slice_sum + remainder_slice)
            if frame.active_rows is not None and frame.active_rows[0] < int(selected_k3_21.shape[1]):
                selected_k3_21 = pad_axis(selected_k3_21[:, :frame.active_rows[0]], 0, int(selected_k3_21.shape[1]) - frame.active_rows[0], 1)
            if far_column is not None:
                selected_k3_21 = fnp.concatenate([selected_k3_21, ONE_THIRD * far_column], axis=1)
            slice_columns = int(selected_k3_21.shape[1])
            if frame.pair_window is not None:
                pieces = [] if pair_first == 0 else [fnp.zeros((pair_first, slice_columns), dtype=fnp.float32)]
                pieces.append(selected_k3_21)
                if pair_last < frame.row_space:
                    pieces.append(fnp.zeros((frame.row_space - pair_last, slice_columns), dtype=fnp.float32))
                frame.k3_21 = fnp.concatenate(pieces, axis=0) if len(pieces) > 1 else selected_k3_21
            elif frame.pair_inverse_order is None:
                frame.k3_21 = selected_k3_21
            else:
                ordered_k3_21 = pad_axis(selected_k3_21, 0, frame.row_space - state.pair_kink_rows, 0)
                frame.k3_21 = ordered_k3_21[frame.pair_inverse_order, :]
            if frame.row_inverse is not None:
                if frame.row_space < state.width:
                    frame.k3_21 = pad_axis(frame.k3_21, 0, state.width - frame.row_space, 0)
                frame.k3_21 = frame.k3_21[frame.row_inverse, :]
                if far_column is not None:
                    kept_columns = slice_columns - 1
                    pieces = [frame.k3_21[:, :kept_columns], frame.k3_21[:, kept_columns:slice_columns] * fnp.ones((1, frame.row_space - kept_columns), dtype=fnp.float32)]
                    if state.width > frame.row_space:
                        pieces.append(fnp.zeros((state.width, state.width - frame.row_space), dtype=fnp.float32))
                    frame.k3_21 = fnp.concatenate(pieces, axis=1)
                elif slice_columns < state.width:
                    frame.k3_21 = pad_axis(frame.k3_21, 0, state.width - slice_columns, 1)
                frame.k3_21 = frame.k3_21[:, frame.row_inverse]
            frame.k3_21 = zero_diagonal_owned(frame.k3_21)
            if frame.pair_window is not None:
                if pair_first > 0:
                    far_ids = fnp.concatenate([frame.row_keep[:pair_first], frame.row_keep[pair_last:]], axis=0)
                    far_diagonal = fnp.concatenate([row_space_k3_diagonal[:pair_first], row_space_k3_diagonal[pair_last:]], axis=0)
                else:
                    far_ids = frame.row_keep[pair_last:]
                    far_diagonal = row_space_k3_diagonal[pair_last:]
                frame.far_moment_data = (frame.pair_window, moment_groups, far_diagonal, frame.row_keep, far_ids)
            elif frame.pair_selected is not None:
                far_rows = frame.pair_order[state.pair_kink_rows:]
                frame.far_moment_data = (far_rows, moment_groups, row_space_k3_diagonal[far_rows], frame.row_keep, far_rows if frame.row_keep is None else frame.row_keep[far_rows])
        frame.pre_transport = state.zeros_vector
        if state.carrier_terms:
            k3_level = fnp.mean(frame.k3_diagonal)
            k3_centred = frame.k3_diagonal - k3_level
            k3_energy = fnp.einsum('i,i->', k3_centred, k3_centred, optimize='optimal')
            for (channel, coefficient) in state.carrier_terms:
                layer_coefficient = coefficient
                raw = frame.mean_pre * frame.variance_pre
                raw = raw - fnp.mean(raw)
                raw_energy = fnp.einsum('i,i->', raw, raw, optimize='optimal')[None]
                gain = fnp.sqrt(k3_energy / fnp.maximum(raw_energy, VAR_FLOOR))
                frame.k3_diagonal = frame.k3_diagonal + layer_coefficient * (gain * raw - k3_centred)
    if frame.star_channel is not None and frame.family_channel is not None:
        frame.star_channel = frame.star_channel + frame.family_channel
    if frame.terminal_inverse_order is not None and (not (frame.is_final and (not frame.expand_terminal_births))):
        frame.k3_diagonal = fnp.where(frame.terminal_inverse_order < state.terminal_kink_rows, frame.k3_diagonal, state.zeros_vector)

def calibrate_k3_level(state, frame):
    layer = frame.layer
    if 1 <= layer <= 14:
        _r_k3_level = float(PARAMS['r_k3band'][0 if layer <= 5 else 1 if layer <= 10 else 2])
        if _r_k3_level != 0.0:
            frame.k3_diagonal = frame.k3_diagonal * (1.0 + _r_k3_level)

def prepare_source_birth(state, frame):
    runtime = state.runtime
    layer = frame.layer
    frame.w31 = frame.wick[W31]
    frame.w01 = frame.wick[W01]
    if frame.pre_transport is None:
        frame.path_repeated = fnp.einsum('i,ij->ij', frame.w11, frame.covariance_offdiag, optimize=False)
    else:
        ratio = -frame.mean_pre / frame.variance_floor
        frame.path_repeated = newborn_path(runtime, frame.covariance_offdiag, frame.k3_21, frame.w11, frame.w21, ratio)
    _fe45_ratio = -frame.mean_pre / frame.variance_floor if frame.pre_transport is None else ratio
    frame.path_repeated = frame.path_repeated + path_tree(runtime, frame.covariance_offdiag, _fe45_ratio, frame.w21, layer)
    _w21t = frame.w21 + fnp.einsum('z,i,i->i', runtime.coefficients['gain3'], frame.k3_diagonal, frame.wick[W51], optimize=False)
    state.carried_gain = frame.w11
    if layer >= 1:
        state.carried_gain = carried_gain(runtime, frame.w11, frame.k3_diagonal, frame.k4_diagonal, frame.wick[W41], frame.wick[W51], frame.w71)
    path_21 = fnp.einsum('ij,ij,j->ij', frame.path_repeated, frame.path_repeated, _w21t, optimize=False)
    if frame.pre_transport is None:
        p111_diagonal = state.zeros_vector
        p111_21 = path_21
    else:
        _carried_gain_squared = state.carried_gain * state.carried_gain
        p111_diagonal = fnp.einsum('i,i,i->i', _carried_gain_squared, state.carried_gain, frame.k3_diagonal, optimize=False)
        p111_21 = fnp.einsum('ij,i,j->ij', frame.k3_21, _carried_gain_squared, state.carried_gain, optimize=False) + path_21
    correction_diagonal = frame.k3_out_diagonal - p111_diagonal
    correction_21 = frame.k3_out_21 - p111_21
    correction_singleton = fnp.einsum('z,ji->ij', runtime.coefficients['three'], correction_21, optimize=False)
    fnp.fill_diagonal(correction_singleton, correction_diagonal)
    frame.u_first = frame.v01 - (frame.p1 * frame.v11 if frame.p1_v11 is None else frame.p1_v11)
    reduced_singleton = fnp.einsum('z,ij,j->ij', runtime.coefficients['six'], frame.path_repeated, frame.u_first, optimize=False)
    frame.free_diagonal = correction_diagonal - fnp.diag(reduced_singleton)
    fnp.fill_diagonal(reduced_singleton, correction_diagonal)
    remainder = correction_singleton - reduced_singleton
    (frame.remainder_left, frame.remainder_right) = remainder_factors(remainder, state.remainder_projection)
    _fe52_reuse = runtime.active_rows[0] is not None and runtime.active_rows[0][0] == layer
    frame.alive_count = None
    _fe10_rs = state.width if frame.row_keep is None else int(frame.row_keep.shape[0])
    if not _fe52_reuse and layer >= 5:
        _fe10_sd2 = frame.gaussian[1]
        frame.alive_count = count_alive(frame.mean_pre > PARAMS['active_alpha_cut'] * _fe10_sd2, 64, min(_fe10_rs, state.width - 32))
    if frame.alive_count is not None and frame.alive_count > _fe10_rs:
        frame.alive_count = _fe10_rs
    if frame.alive_count == state.width:
        frame.alive_count = None
    if _fe52_reuse:
        frame.alive_count = runtime.active_rows[0][1]
    if not _fe52_reuse and frame.alive_count is not None and (frame.row_order is not None):
        _v144_rule = (PARAMS['active_tail_epsilon'], 256)
        frame.alive_count = tail_count(fnp.abs(frame.w21[frame.row_order]), int(frame.alive_count), float(_v144_rule[0]), 32, int(_v144_rule[1]), 64)
    transport_count = rank_at(state.transport_tiers, layer, state.width)
    if transport_count is not None and frame.alive_count is not None and (transport_count > frame.alive_count):
        transport_count = frame.alive_count
    if transport_count is None:
        transport_count = frame.alive_count
    if frame.alive_count is None and transport_count is None:
        alive_order = None
    elif frame.row_order is not None:
        alive_order = frame.row_order
    else:
        alive_key = -frame.w11
        alive_order = fnp.argsort(alive_key, stable=True)
    frame.transport_alive = None if transport_count is None else alive_order[:transport_count]
    if frame.alive_count is None:
        frame.alive = None
        frame.singleton_diagonal = 3.0 * _w21t
    else:
        frame.alive = alive_order[:frame.alive_count]
        frame.path_repeated = frame.path_repeated[frame.alive, :][:, frame.alive]
        frame.u_first = frame.u_first[frame.alive]
        frame.free_diagonal = frame.free_diagonal[frame.alive]
        frame.remainder_left = frame.remainder_left[frame.alive, :]
        frame.remainder_right = frame.remainder_right[frame.alive, :]
        frame.singleton_diagonal = 3.0 * _w21t[frame.alive]

def store_source_birth(state, frame):
    frame.free_diagonal = frame.free_diagonal - fnp.einsum('ij,ij->i', frame.remainder_left, frame.remainder_right, optimize='optimal')
    _fe10_qo = 0
    if frame.alive is not None:
        _fe10_n = int(frame.alive.shape[0])
        _fe10_ta = frame.mean_pre[frame.alive] / fnp.sqrt(frame.variance_floor[frame.alive])
        _fe10_qo = int(fnp.sum(_fe10_ta > PARAMS['source_prefix_alpha_cut'])) // 64 * 64
        _fe10_qo = max(0, min(_fe10_qo, _fe10_n - 64))
        if _fe10_qo > 0:
            _fe10_m = pad_axis(fnp.ones(_fe10_n - _fe10_qo, dtype=frame.singleton_diagonal.dtype), _fe10_qo, 0, 0)
            frame.path_repeated = frame.path_repeated[:, _fe10_qo:]
            frame.singleton_diagonal = frame.singleton_diagonal * _fe10_m
            frame.u_first = frame.u_first * _fe10_m
            if state.source_k4_star_vertices is not None:
                state.source_k4_star_vertices[-1] = state.source_k4_star_vertices[-1] * _fe10_m
            if state.source_k4_family_vertices is not None:
                _fe10_fv = list(state.source_k4_family_vertices[-1])
                for _fe10_j in (0, 1):
                    if _fe10_fv[_fe10_j] is not None:
                        _fe10_fv[_fe10_j] = _fe10_fv[_fe10_j] * _fe10_m
                state.source_k4_family_vertices[-1] = tuple(_fe10_fv)
    state.source_path_factors.append(frame.path_repeated)
    state.source_free_u.append(frame.u_first)
    state.source_free_diagonals.append(frame.free_diagonal)
    state.source_remainder_left.append(frame.remainder_left)
    state.source_remainder_right.append(frame.remainder_right)
    state.source_singleton_diagonals.append(frame.singleton_diagonal)
    state.birth_self_c.append(frame.w11 * frame.variance_floor if frame.alive is None else (frame.w11 * frame.variance_floor)[frame.alive])
    state.birth_c2.append(frame.w21 if frame.alive is None else frame.w21[frame.alive] if _fe10_qo <= 0 else frame.w21[frame.alive] * _fe10_m)
    state.source_alive.append(frame.alive)
    state.source_transport_alive.append(frame.transport_alive)
    state.q_offsets.append(_fe10_qo)
