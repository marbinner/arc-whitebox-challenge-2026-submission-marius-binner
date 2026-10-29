"""Stable row selection and calibrated cumulant tail restoration.
Inputs: derivatives, means, variances, row IDs and tier strings. Outputs: selected
domains and filled tails. Knobs: active cuts, tiers and pair/terminal row counts."""
from __future__ import annotations
from .array_ops import pad_axis, reverse_vector
from .config import PARAMS
from .linalg import _fast_matmul
from .state import W21, fnp

def select_active_rows(runtime, layer, is_final, width, mean_pre, gaussian, wick, row_keep, row_order, col_count, pair_window, pair_selected):
    runtime.active_rows[0] = None
    if is_final or row_keep is None or row_order is None or (int(width) != 1024) or (layer < 5):
        return None
    (lo, hi) = (5, 14)
    if not int(lo) <= layer <= int(hi):
        return None
    rs = int(row_keep.shape[0])
    sd = gaussian[1]
    alive_count = count_alive(mean_pre > PARAMS['active_alpha_cut'] * sd, 64, min(rs, width - 32))
    if alive_count > rs:
        alive_count = rs
    if alive_count == width:
        runtime.active_rows[0] = (int(layer), None)
        return None
    rule = (PARAMS['active_tail_epsilon'], 256)
    alive_count = tail_count(fnp.abs(wick[W21][row_order]), int(alive_count), float(rule[0]), 32, int(rule[1]), 64)
    runtime.active_rows[0] = (int(layer), int(alive_count))
    R = int(alive_count)
    if col_count is not None:
        R = max(R, int(col_count))
    if pair_window is not None:
        R = max(R, int(pair_window[1]))
    elif pair_selected is not None:
        R = rs
    R = (R + 64 - 1) // 64 * 64
    if R >= rs:
        return None
    return (R, int(alive_count))

def fill_active_rows(rows, k3, star, fam, acc, sigma_rk):
    R = rows[0]
    rs = int(k3.shape[0])
    if R >= rs:
        return (k3, star, fam, acc)
    s2 = sigma_rk * sigma_rk
    s3 = s2 * sigma_rk
    s4 = s2 * s2
    vecs = [k3]
    pows = [s3]
    slots = [('k3', None)]
    for (name, v) in (('star', star), ('fam', fam)):
        if v is not None:
            vecs.append(v)
            pows.append(s4)
            slots.append((name, None))
    if acc:
        for (i, item) in enumerate(acc):
            for (j, x) in enumerate(item):
                vecs.append(x)
                pows.append(s2)
                slots.append(('acc', (i, j)))
    lo = max(0, R - 64)
    V = fnp.stack(vecs, axis=0)
    P = fnp.stack(pows, axis=0)
    Z = V[:, lo:R] / P[:, lo:R]
    ratio = fnp.mean(Z, axis=1)
    tail = ratio[:, None] * P[:, R:]
    out = fnp.concatenate([V[:, :R], tail], axis=1)
    filled = fnp.unstack(out, axis=0)
    acc = [list(item) for item in acc] if acc else acc
    for ((name, key), v) in zip(slots, filled):
        if name == 'k3':
            k3 = v
        elif name == 'star':
            star = v
        elif name == 'fam':
            fam = v
        else:
            acc[key[0]][key[1]] = v
    acc = [tuple(item) for item in acc] if acc else acc
    return (k3, star, fam, acc)

def terminal_fill_setup(order, tkr, mean_pre, sigma):
    if tkr >= int(order.shape[0]):
        return None
    mu = mean_pre[order]
    sg = sigma[order]
    pos = (mu > 0.0).astype(fnp.float32)
    return (tkr, 64, pos, sg)

def terminal_fill(sel, tf, power):
    (tkr, w, pos, sg) = tf
    s2 = sg * sg
    sp = s2 * sg if power == 3 else s2 * s2 if power == 4 else s2
    lo = max(0, tkr - w)
    z = sel[lo:tkr] / sp[lo:tkr]
    pw = pos[lo:tkr]
    r_pos = fnp.sum(z * pw) / (fnp.sum(pw) + 1e-06)
    negative_weight = 1.0 - pw
    r_neg = fnp.sum(z * negative_weight) / (fnp.sum(negative_weight) + 1e-06)
    tail_pos = pos[tkr:]
    tail = (r_neg + (r_pos - r_neg) * tail_pos) * sp[tkr:]
    return fnp.concatenate([sel, tail], axis=0)

def pad_product_rows(mp, top, nrows):
    return pad_axis(mp, top, max(0, nrows - top - int(mp.shape[1])), 1)

def active_rows_matmul(left, right, rows):
    if rows is None or rows[0] >= int(left.shape[-2]):
        return _fast_matmul(left, right, leg_transport=True)
    R = rows[0]
    top = _fast_matmul(left[:R, :], right, leg_transport=True)
    return pad_axis(top, 0, int(left.shape[-2]) - R, 0)

def zero_inactive_rows(a, rows):
    if rows is None or rows[0] >= int(a.shape[0]):
        return a
    return pad_axis(a[:rows[0], :], 0, int(a.shape[0]) - rows[0], 0)

def tail_count(w, n, eps, step, max_drop, floor):
    # w is a nonnegative derivative magnitude. Compute all admissible tails
    # together and transfer only the final index; activation arithmetic stays
    # metered, including the float64 threshold and comparisons.
    drops = max(0, min(max_drop, n - floor) // step)
    if drops == 0:
        return n
    cumulative = fnp.cumsum(reverse_vector(w[:n]), dtype=fnp.float64)
    tails = cumulative[step - 1:drops * step:step]
    total = cumulative[-1]
    count = fnp.searchsorted(tails, fnp.multiply(eps, total), side='right')
    return n - step * int(fnp.multiply(count, total > 0.0))

def select_covariance_rows(alpha, layer, depth, width):
    if width != 1024 or depth != 16 or layer + 1 < 6 or (layer + 1 >= depth):
        return None
    rnd = 128 if layer + 2 == depth else 32
    count = int(fnp.sum(alpha > PARAMS['covariance_alpha_cut']))
    count = max((count + rnd - 1) // rnd * rnd, 512)
    if count >= width:
        return None
    order = fnp.argsort(alpha, stable=True)
    return (order[width - count:], order[:width - count])

def count_alive(flags, lo, hi):
    count = int(fnp.sum(flags))
    count = (count + 32 - 1) // 32 * 32
    return int(max(min(lo, hi), min(hi, count)))

def contiguous_pair_window(ids):
    """Certify a contiguous set of distinct row IDs from an argsort prefix."""
    count = int(ids.shape[0])
    if count == 0:
        return None
    first = int(fnp.min(ids))
    last = int(fnp.max(ids))
    return (first, first + count) if last - first == count - 1 else None

def select_transport(weight, carried_gain, transport_alive, terminal_selected, expand_final, row_keep):
    if transport_alive is None:
        gained_weight = weight
        gained_gain = carried_gain
    else:
        gained_weight = weight[transport_alive, :]
        gained_gain = carried_gain[transport_alive]
    if terminal_selected is not None and (not expand_final):
        gained_weight = gained_weight[:, terminal_selected]
    if row_keep is not None:
        gained_weight = gained_weight[:, row_keep]
    return (gained_weight, gained_gain)

def parse_tiers(text: str) -> tuple[tuple[int, int], ...]:
    tiers = []
    for raw_item in str(text).split(','):
        item = raw_item.strip()
        if not item:
            continue
        fields = item.split(':')
        if len(fields) != 2:
            raise ValueError('leg_tiers entries must have form age:rank')
        age = int(fields[0])
        rank = int(fields[1])
        if age <= 0 or rank <= 0:
            raise ValueError('leg_tiers ages and ranks must be positive')
        tiers.append((age, rank))
    tiers.sort()
    for (index, (age, rank)) in enumerate(tiers):
        if index and age == tiers[index - 1][0]:
            raise ValueError('leg_tiers ages must be unique')
        if index and rank > tiers[index - 1][1]:
            raise ValueError('leg_tiers ranks must be non-increasing')
    return tuple(tiers)

def rank_at(tiers: tuple[tuple[int, int], ...], age: int, width: int) -> int | None:
    rank = None
    for (tier_age, tier_rank) in tiers:
        if age >= tier_age:
            rank = min(tier_rank, width)
    return rank if rank is not None and rank < width else None

def restore_terminal_leg_rows(value, inverse, rows, axis):
    """Restore a freshly computed terminal row union; discarded rows stay zero."""
    if inverse is None:
        return value
    value = pad_axis(value, 0, rows - int(value.shape[axis]), axis)
    return value[inverse, :] if axis == 0 else value[:, inverse, :]
