"""Birth contractions H12, carried ages three to six and FEED folding.
Math: birth=(H+remainder)/3; carried fold=-gain*mean/variance; FEED folds Q legs.
Inputs: source legs, recent births and runtime caches. Outputs: slices, diagonal
features, folded Q and next history. Knobs: H12, c_Yi/c_Yii, FEED and fe30 coefficients."""
from __future__ import annotations
from .array_ops import prefix_view, feature_sum, history_rows, pad_axis, q_columns, q_front, restore_history_rows
from .config import PARAMS
from .linalg import _fast_matmul
from .state import ONE_THIRD, VAR_FLOOR, fnp

def apply_feed_fold(runtime, leg_q_dense, qoffs, w21, carried_gain, row_order, read_inverse):
    ratio = w21 / fnp.maximum(carried_gain, 1e-06)
    if row_order is not None:
        ratio = ratio[row_order]
    for (source, H, row_ids, beta, vids) in runtime.feed_pending:
        old_q = leg_q_dense[source]
        if old_q is None:
            continue
        rows = int(old_q.shape[0])
        qoff = int(qoffs[source])
        full_columns = qoff + int(old_q.shape[1])
        columns = int(H.shape[1])
        if columns > full_columns:
            raise ValueError('feed_fold H exceeds source vertex domain')
        rr = ratio if int(ratio.shape[0]) == rows else prefix_view(ratio, rows)
        rr = history_rows(rr[:, None], row_ids)
        delta = float(beta) * rr * H
        if columns < full_columns:
            delta = pad_axis(delta, 0, full_columns - columns, 1)
        delta = q_columns(delta, qoff)
        delta = restore_history_rows(delta, row_ids, read_inverse, rows)
        leg_q_dense[source] = old_q + delta
    runtime.feed_pending.clear()

def history_select(a, state, cache=None):
    ids = state['transport']
    if state['row_inverse'] is None:
        if ids is None:
            return a
        selector = ('indexed', id(ids))
        index = (ids, slice(None))
    else:
        n = int(a.shape[0]) if ids is None else int(ids.shape[0])
        if n >= int(a.shape[0]):
            return a
        selector = ('prefix', n)
        index = (slice(None, n), slice(None))
    if cache is None:
        return a[index]
    key = ('history_select', id(a), selector)
    found = cache.get(key)
    if found is None:
        found = (a, ids, a[index])
        cache[key] = found
    return found[2]

def history_factors(first, second, rem_left, ids, cache=None):
    (f, s) = (history_rows(first, ids, cache), history_rows(second, ids, cache))
    return (fnp.concatenate([f, s], axis=1), rem_left)

def birth_q_front(runtime, part, qoff, width):
    cache = runtime.readout['q_views']
    key = (id(part), int(qoff), int(width))
    wide = runtime.readout.get('wide_q_front', {}).get(id(part))
    if qoff > 0 and key not in cache and wide is not None and wide[0] is part and wide[1] == qoff and qoff + int(part.shape[1]) <= width <= wide[2]:
        cache[key] = (part, prefix_view(wide[3], width, 1))
    return q_front(part, qoff, width, cache)

def birth_readout_rows(runtime, source, t, q, self_c, read_ids, vertex_ids, width):
    found = runtime.readout.get('birth_operands', {}).get(source)
    if found is None or found[0] is not t[source] or found[1] is not q[source] or found[2] is not self_c[source] or found[3] is not read_ids or found[4] is not vertex_ids:
        return None
    if vertex_ids is None:
        if width > int(found[5].shape[1]):
            return None
        return (prefix_view(found[5], width, 1), prefix_view(found[6], width, 1))
    return (found[5][:, vertex_ids], found[6][:, vertex_ids])

def birth_chain(runtime, first, second, t, q, left, right, path, rem_right, rem_left, history, row_ids, row_keep, col_count, row_space, layer, terminal, self_c, c2, mean_pre, variance, blocks_selected=False):
    if blocks_selected and not terminal:
        raise ValueError('already selected birth blocks require the terminal chain')
    old = len(t) - 1
    block_rows = None if blocks_selected else row_ids
    factors = [None for _ in t]
    for i in range(len(t)):
        if right[i] is None and (not terminal or i < old):
            factors[i] = history_factors(first[i], second[i], rem_left[i], block_rows, runtime.readout['history_views'])
    _v144_terminal_chain = terminal and layer >= 3 and (len(history) >= 2)
    shared_groups = {}
    for i in range(len(t)):
        if right[i] is not None:
            shared_groups.setdefault(id(left[i]), []).append(i)
    aggregate_ids = set()
    _f38_inj = joined_history_context(runtime, history, right, terminal, layer, first, second, path)
    if _f38_inj:
        for i in _f38_inj:
            factors[i] = (None, None)
        aggregate_ids.update(_f38_inj)
    if not _v144_terminal_chain:
        for indices in shared_groups.values():
            indices = [i for i in indices if i not in _f38_inj]
            if len(indices) < 2:
                continue
            aa = []
            bb = []
            for i in indices:
                aa.extend([history_rows(first[i], block_rows, runtime.readout['history_views']), history_rows(second[i], block_rows, runtime.readout['history_views'])])
                bb.extend([path[i], right[i]])
            A = fnp.concatenate(aa, axis=1)
            B = fnp.concatenate(bb, axis=0)
            thin = _fast_matmul(A, B)
            factors[indices[0]] = (thin, None)
            for i in indices[1:]:
                factors[i] = (None, None)
            aggregate_ids.update(indices)
    _skip_joined_rows = terminal and layer >= 3 and (len(history) >= 2)
    keys = [] if _skip_joined_rows else sorted({(tuple(first[i].shape[1:]), tuple(second[i].shape[1:]), tuple(right[i].shape), tuple(path[i].shape)) for i in range(len(t)) if right[i] is not None and i not in aggregate_ids}, reverse=True)
    for key in keys:
        indices = [i for i in range(len(t)) if right[i] is not None and i not in aggregate_ids and ((tuple(first[i].shape[1:]), tuple(second[i].shape[1:]), tuple(right[i].shape), tuple(path[i].shape)) == key)]
        xy = fnp.stack([fnp.concatenate([history_rows(first[i], block_rows, runtime.readout['history_views']), history_rows(second[i], block_rows, runtime.readout['history_views'])], axis=1) for i in indices], axis=0)
        cb = fnp.stack([fnp.concatenate([path[i], right[i]], axis=0) for i in indices], axis=0)
        thin = _fast_matmul(xy, cb)
        thin_rows = fnp.unstack(thin, axis=0) if len(indices) > 1 else None
        for (j, i) in enumerate(indices):
            factors[i] = (thin_rows[j] if thin_rows is not None else thin[j], None)

    def columns(i, state):
        if right[i] is not None:
            return history_select(state['left'][i], state, runtime.readout['history_views'])
        pt = history_select(state['t'][i], state, runtime.readout['history_views'])
        pq = history_select(state['q'][i], state, runtime.readout['history_views'])
        return fnp.concatenate([pq, pt], axis=1)
    return carried_birth_chain(runtime, factors, columns, t, q, left, right, rem_right, rem_left, history, row_ids, row_keep, col_count, row_space, layer, terminal, self_c, c2, mean_pre, variance, old)

def consume_births(runtime, hs, t, q, self_c, c2, read_ids, read_inverse, row_inverse, row_space, width):
    D = None
    _f31_he = []
    for (source, H, vertex_ids) in hs:
        v = int(H.shape[1])
        cached = birth_readout_rows(runtime, source, t, q, self_c, read_ids, vertex_ids, v)
        if cached is None:
            M = prefix_view(t[source], v, 1) if vertex_ids is None else t[source][:, vertex_ids]
            _fe10_qf = birth_q_front(runtime, q[source], int(t[source].shape[1]) - int(q[source].shape[1]), int(t[source].shape[1]))
            Q = prefix_view(_fe10_qf, v, 1) if vertex_ids is None else _fe10_qf[:, vertex_ids]
            sc = prefix_view(self_c[source], v) if vertex_ids is None else self_c[source][vertex_ids]
            Q = Q + fnp.einsum('ij,j->ij', M, sc, optimize=False)
            mr = history_rows(M, read_ids, runtime.readout['history_views'])
            qr = history_rows(Q, read_ids, runtime.readout['history_views'])
        else:
            mr, qr = cached
        cc = prefix_view(c2[source], v) if vertex_ids is None else c2[source][vertex_ids]
        wh = fnp.einsum('ij,j->ij', H, cc, optimize=False)
        d = fnp.einsum('z,ij,ij,ij->i', runtime.coefficients['twelve'], wh, mr, qr, optimize=False)
        _f31_he.append(birth_features(wh, mr, qr))
        D = d if D is None else D + d
    D = restore_history_rows(D, read_ids, read_inverse, row_space)
    if row_inverse is not None:
        if row_space < width:
            D = pad_axis(D, 0, width - row_space, 0)
        D = D[row_inverse]
    runtime.features['hrow'] = finish_birth_features(_f31_he, read_ids, read_inverse, row_space, row_inverse, width)
    return D

def carried_birth_chain(runtime, factors, columns, t, q, left, right, rem_right, rem_left, history, row_ids, row_keep, col_count, row_space, layer, terminal, self_c, c2, mean_pre, variance, old):
    state1 = history[-1]
    state2 = history[-2]
    src2 = [i for i in range(old - 1) if not right[i] is not None]
    src1 = src2 + [old - 1]
    _fe33_H = {}
    _fe33_A = 0
    _f38 = None if terminal else runtime.history_join['ctx']
    if not terminal and layer >= 10:
        _fe33_A = min(6, len(history), old)
        while _fe33_A > 2 and (not [i for i in range(old - _fe33_A + 1) if not right[i] is not None]):
            _fe33_A -= 1
        if any((right[old - _a + 1] is not None for _a in range(3, _fe33_A + 1))):
            _fe33_A = 0
    if _f38 is not None:
        plan_joined_history(_f38, _fe33_A, history, row_ids)
    if _fe33_A >= 3:
        _fe33_src = [i for i in range(old - _fe33_A + 1) if not right[i] is not None]
        _F38B = joined_history_blocks(_f38, _fe33_A)
        _fe33_HA = _fast_matmul(fnp.concatenate([factors[i][0] for i in _fe33_src] + _F38B[0], axis=1), fnp.concatenate([columns(i, history[-_fe33_A]) for i in _fe33_src] + _F38B[1], axis=1).T)
        _fe33_H[_fe33_A] = (_fe33_HA, list(_fe33_src) + joined_history_remainders(_f38, _fe33_A))
        for _a in range(_fe33_A, 2, -1):
            _s = old - _a + 1
            _sa = history[-_a]
            _ga = _sa['gain'] if _sa['transport'] is None else _sa['gain'][_sa['transport']]
            _na = int(_fe33_HA.shape[1])
            _ks = int(t[_s].shape[1])
            _nqs = int(q[_s].shape[1])
            _car = fnp.einsum('ij,j->ij', _fe33_HA, _ga, optimize=False)
            if _na < _ks:
                _car = pad_axis(_car, 0, _ks - _na, 1)
            _Fs = factors[_s][0]
            _cb = columns(_s, history[-(_a - 1)])
            _F38B = joined_history_blocks(_f38, _a - 1)
            _fe33_HA = _fast_matmul(fnp.concatenate([_Fs[:, _nqs:] + _car, _Fs[:, :_nqs]] + _F38B[0], axis=1), fnp.concatenate([_cb[:, _nqs:], _cb[:, :_nqs]] + _F38B[1], axis=1).T)
            _fe33_src = _fe33_src + [_s]
            _fe33_H[_a - 1] = (_fe33_HA, list(_fe33_src) + joined_history_remainders(_f38, _a - 1))
        H2 = _fe33_HA
    else:
        _F38B = joined_history_blocks(_f38, 2)
        H2 = _fast_matmul(fnp.concatenate([factors[i][0] for i in src2] + _F38B[0], axis=1), fnp.concatenate([columns(i, state2) for i in src2] + _F38B[1], axis=1).T)
    n2 = int(H2.shape[1])
    k2 = int(t[old - 1].shape[1])
    nq2 = int(q[old - 1].shape[1])
    row_rem = {}
    proj = {}

    def hrem(ids, state):
        rl = []
        rr = []
        for i in ids:
            if i not in row_rem:
                if not terminal and runtime.remainder_products:
                    row_rem[i] = runtime.remainder_rows[i]
                else:
                    ti = history_rows(t[i], row_ids, runtime.readout['history_views'])
                    row_rem[i] = _fast_matmul(ti * ti, rem_right[i])
            rl.append(row_rem[i])
            _fe11_tl = state.get('tl')
            if _fe11_tl is not None and (right[i] is None or i not in state['dense_ids'] or (_f38 is not None and i in _f38['age'])):
                _fe11_rows = state['tl_rows']
                if i not in _fe11_rows:
                    _fe11_rows[i] = history_select(_fe11_tl[i], state, runtime.readout['history_views'])
                rr.append(_fe11_rows[i])
            elif right[i] is None:
                rr.append(_fast_matmul(history_select(state['t'][i], state, runtime.readout['history_views']), rem_left[i]))
            else:
                if i not in proj:
                    proj[i] = _fast_matmul(right[i].T, rem_left[i])
                rr.append(_fast_matmul(history_select(state['left'][i], state, runtime.readout['history_views']), proj[i]))
        return _fast_matmul(fnp.concatenate(rl, axis=1), fnp.concatenate(rr, axis=1).T)
    H2r = (H2 + hrem(src2 + joined_history_remainders(_f38, 2), state2)) * ONE_THIRD
    fold = not terminal
    if fold:
        scale = -float(PARAMS['h12_fold_gain']) * mean_pre / fnp.maximum(variance, VAR_FLOOR)
        if row_keep is not None:
            scale = scale[row_keep]
        scale = history_rows(scale[:, None], row_ids, runtime.readout['history_views'])
        group = 0 if layer <= 5 else 1 if layer <= 10 else 2
        ci = float(PARAMS['c_Yi'][group])
        cii = float(PARAMS['c_Yii'][group])

    def fold_delta(source, H, vertex_ids=None):
        v = int(H.shape[1])
        k = int(t[source].shape[1])
        sc = prefix_view(self_c[source], v)
        M = history_rows(prefix_view(t[source], v, 1), row_ids, runtime.readout['history_views'])
        Q = history_rows(prefix_view(birth_q_front(runtime, q[source], k - int(q[source].shape[1]), k), v, 1), row_ids, runtime.readout['history_views']) + fnp.einsum('ij,j->ij', M, sc, optimize=False)
        runtime.readout.setdefault('birth_operands', {})[source] = (t[source], q[source], self_c[source], row_ids, vertex_ids, M, Q)
        wh = fnp.einsum('ij,j->ij', H, prefix_view(c2[source], v), optimize=False)
        df = scale * ci * wh * M
        ds = scale * cii * wh * Q + fnp.einsum('ij,j->ij', df, sc, optimize=False)
        _fe10_qo = k - int(q[source].shape[1])
        if v < k:
            pad = fnp.zeros((int(H.shape[0]), k - v), dtype=fnp.float32)
            return fnp.concatenate([q_columns(fnp.concatenate([df, pad], axis=1), _fe10_qo), ds, pad], axis=1)
        return fnp.concatenate([q_columns(df, _fe10_qo), ds], axis=1)
    F2 = factors[old - 1][0]
    H2m = H2r
    keep = max(1, int(n2 * 0.5))
    perm = fnp.argsort(-fnp.abs(prefix_view(c2[old - 1], n2)), stable=True)
    vids = perm[:keep]
    if fold:
        vmask = pad_axis(fnp.ones((keep,), dtype=fnp.float32), 0, n2 - keep, 0)[fnp.argsort(perm, stable=True)]
        H2m = fnp.einsum('ij,j->ij', H2r, vmask, optimize=False)
    if fold:
        F2 = F2 + fold_delta(old - 1, H2m, vids)
    g2 = state2['gain'] if state2['transport'] is None else state2['gain'][state2['transport']]
    carried2 = fnp.einsum('ij,j->ij', H2, g2, optimize=False)
    if n2 < k2:
        carried2 = pad_axis(carried2, 0, k2 - n2, 1)
    cols1 = columns(old - 1, state1)
    _F38B = joined_history_blocks(_f38, 1)
    H1 = _fast_matmul(fnp.concatenate([F2[:, nq2:] + carried2, F2[:, :nq2]] + _F38B[0], axis=1), fnp.concatenate([cols1[:, nq2:], cols1[:, :nq2]] + _F38B[1], axis=1).T)
    H1r = (H1 + hrem(src1 + joined_history_remainders(_f38, 1), state1)) * ONE_THIRD
    if runtime.feed_active and (not terminal):
        (_feed_b1, _feed_b2) = PARAMS['feed_fold_coeffs'][layer]
        if float(_feed_b2) != 0.0:
            _feed_H2 = H2m
            runtime.feed_pending.append((old - 1, _feed_H2, row_ids, _feed_b2, vids))
        if float(_feed_b1) != 0.0:
            runtime.feed_pending.append((old, H1r, row_ids, _feed_b1, None))
    readouts = [(old - 1, H2r[:, vids], vids), (old, H1r, None)]
    for _a in (3, 4, 5, 6) if _fe33_H else ():
        if _a not in _fe33_H:
            continue
        (_Hc, _sc) = _fe33_H[_a]
        _sn = old - _a + 1
        _nA = int(_Hc.shape[1])
        _hr = hrem(_sc, history[-_a])
        _keep = max(1, int(_nA * 0.5))
        _vids = fnp.argsort(-fnp.abs(prefix_view(c2[_sn], _nA)), stable=True)[:_keep]
        runtime.older_births.append((_sn, (_Hc[:, _vids] + _hr[:, _vids]) * ONE_THIRD, _vids))
    if terminal:
        return (None, None, readouts)
    F1 = factors[old][0]
    if fold:
        F1 = F1 + fold_delta(old, H1r)
    k1 = int(t[old].shape[1])
    nq1 = int(q[old].shape[1])
    v1 = int(H1.shape[1])
    g1 = state1['gain'] if state1['transport'] is None else state1['gain'][state1['transport']]
    carried = fnp.einsum('ij,j->ij', H1, g1, optimize=False)
    if v1 < k1:
        carried = pad_axis(carried, 0, k1 - v1, 1)
    A = fnp.concatenate([carried + F1[:, nq1:], F1[:, :nq1]], axis=1)
    B = fnp.concatenate([t[old], q[old]], axis=1).T
    joined = [i for i in range(old - 1) if right[i] is not None and (not (_f38 is not None and i in _f38['inj']))]
    if joined:
        shared = {}
        for i in joined:
            shared.setdefault(id(left[i]), []).append(i)
        aa = []
        bb = []
        for indices in shared.values():
            combined = factors[indices[0]][0]
            for i in indices[1:]:
                if factors[i][0] is not None:
                    combined = combined + factors[i][0]
            aa.append(combined)
            bb.append(left[indices[0]].T)
        A = fnp.concatenate([A] + aa, axis=1)
        B = fnp.concatenate([B] + bb, axis=0)
    c = row_space if col_count is None else col_count
    result = _fast_matmul(A, prefix_view(B, c, 1))
    far = None if c == row_space else _fast_matmul(A, fnp.mean(B[:, c:], axis=1)[:, None])
    return (result, far, readouts)

def joined_history_context(runtime, history, right, terminal, layer, first, second, path):
    runtime.history_join['ctx'] = None
    if terminal or layer < 3 or len(history) < 2:
        return set()
    inj = set()
    h1 = history[-1]
    for i in range(len(right)):
        if right[i] is None or layer - i < 0 or i >= len(h1['left']):
            continue
        if h1['left'][i] is not None and h1['right'][i] is right[i]:
            inj.add(i)
    runtime.history_join['ctx'] = dict(inj=inj, first=first, second=second, path=path, right=right, blocks={}, age={}, views=runtime.readout['history_views'])
    return inj

def plan_joined_history(ctx, A, history, row_ids):
    ctx['blocks'] = {}
    ctx['age'] = {}
    if not ctx['inj']:
        return
    cap = max(1, min(A if A >= 3 else 2, 99, len(history)))
    (first, second, path, right) = (ctx['first'], ctx['second'], ctx['path'], ctx['right'])
    groups = {}
    for i in sorted(ctx['inj']):
        a_i = None
        for a in range(cap, 0, -1):
            st = history[-a]
            if i >= len(st['left']) or st['left'][i] is None:
                continue
            if not st['right'][i] is right[i]:
                continue
            a_i = a
            break
        if a_i is None:
            raise ValueError('injected source without a usable history state')
        ctx['age'][i] = a_i
        groups.setdefault((a_i, id(history[-a_i]['left'][i])), []).append(i)
    for ((a_i, _), members) in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1][0])):
        st = history[-a_i]
        A_ = fnp.concatenate([x for i in members for x in (history_rows(first[i], row_ids, ctx['views']), history_rows(second[i], row_ids, ctx['views']))], axis=1)
        B_ = fnp.concatenate([x for i in members for x in (st['path'][i], st['right'][i])], axis=0)
        thin = _fast_matmul(A_, B_)
        cols = history_select(st['left'][members[0]], st, ctx['views'])
        blk = ctx['blocks'].setdefault(a_i, ([], []))
        blk[0].append(thin)
        blk[1].append(cols)

def joined_history_blocks(ctx, age):
    if ctx is None or age not in ctx['blocks']:
        return ([], [])
    return ctx['blocks'][age]

def joined_history_remainders(ctx, age):
    if ctx is None:
        return []
    return [i for i in sorted(ctx['age']) if ctx['age'][i] >= age]

def birth_features(wh, mr, qr):
    wq = wh * qr
    wm = wh * mr
    return fnp.stack([fnp.einsum('ij,ij->i', wh, wh), fnp.einsum('ij,ij->i', wq, wq), fnp.einsum('ij,ij->i', wm, wm), fnp.einsum('ij,ij->i', wm, mr)], axis=1)

def finish_birth_features(acc, ids, inverse, rows, row_inverse, width):
    if not acc:
        return None
    out = restore_history_rows(feature_sum(acc), ids, inverse, rows)
    if row_inverse is not None:
        if rows < width:
            out = pad_axis(out, 0, width - rows, 0)
        out = out[row_inverse]
    return out

def apply_birth_diagonal(state, frame):
    runtime = state.runtime
    layer = frame.layer
    if frame.birth_diagonal is not None:
        _h12_bd = float(PARAMS['h12_beta_layers'][layer])
        _h12_gd = float(PARAMS['h12_gamma_layers'][layer])
        frame.radial_birth_input = frame.radial_k4
        frame.birth_delta = _h12_bd * frame.birth_diagonal + _h12_gd * frame.radial_birth_input
        frame.k4_diagonal = frame.k4_diagonal + frame.birth_delta
        if layer in runtime.birth_diagonal:
            runtime.features['dx'] = runtime.birth_diagonal[layer]
            runtime.features['dx_layer'] = layer
            _fe30_c = PARAMS['fe30']['coef'][0 if layer <= 10 else 1]
            frame.k4_diagonal = frame.k4_diagonal + float(_fe30_c) * runtime.birth_diagonal.pop(layer)

def finish_history(state, frame):
    runtime = state.runtime
    layer = frame.layer
    if layer >= 1:
        state.birth_history.append(dict(t=list(frame.suffix_parts), q=list(frame.repeated_parts), left=list(state.leg_left), **{'right': list(state.leg_right), 'path': list(state.leg_path_right)}, gain=state.carried_gain, transport=state.source_transport_alive[layer], row_inverse=frame.row_inverse, tl=frame.groups[0][2] if len(frame.groups) == 1 else None, tl_rows={}, dense_ids=frozenset((_fe11_i for _fe11_i in range(layer) if state.leg_left[_fe11_i] is None))))
        state.birth_history = state.birth_history[-6:]
    if state.feed_active and runtime.feed_pending:
        apply_feed_fold(runtime, state.leg_q_dense, state.q_offsets, frame.w21, state.carried_gain, frame.row_order, frame.pair_inverse_order)
    state.row_inverse_prev = frame.row_inverse
    state.rows.append(state.mean)
