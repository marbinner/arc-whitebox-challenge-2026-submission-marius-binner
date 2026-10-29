"""Array domains, owned diagonal writes and symmetry-preserving sums.
Math: A+A.T, (A+A.T)/2, padding and row restoration in source addition order.
Inputs/outputs: arrays and row IDs. Knobs: none; caches stay with their caller."""
from __future__ import annotations
from .state import fnp

def reverse_vector(vector):
    """A read-only reverse view; the Torch builder lowers unsupported slices."""
    return vector[::-1]

def pad_axis(a, before, after, axis):
    if before == 0 and after == 0:
        return a
    widths = [(0, 0)] * len(a.shape)
    widths[axis] = (before, after)
    return fnp.pad(a, widths, mode='constant')

def layer_band(value, layer):
    if isinstance(value, (list, tuple)) or hasattr(value, 'shape') and len(value.shape) > 0:
        return float(value[0 if layer <= 5 else 1 if layer <= 10 else 2])
    return float(value)

def q_front(part, qoff, width, cache=None):
    if qoff <= 0:
        return part
    if cache is None:
        return pad_axis(part, qoff, max(0, width - qoff - int(part.shape[1])), 1)
    key = (id(part), int(qoff), int(width))
    found = cache.get(key)
    if found is None:
        found = (part, pad_axis(part, qoff, max(0, width - qoff - int(part.shape[1])), 1))
        cache[key] = found
    return found[1]

def q_columns(block, qoff):
    return block if qoff <= 0 else block[:, qoff:]

def q_blocks(blocks, qoffs):
    return [None if b is None else q_columns(b, qoffs[j]) for (j, b) in enumerate(blocks)]

def pair_rows(block, pair_selected, pair_window):
    if pair_window is not None:
        return block[pair_window[0]:pair_window[1], :]
    if pair_selected is None:
        return block
    return block[pair_selected, :]

def column_rows(block, col_count):
    return block if col_count is None else block[:col_count, :]

def far_mean_rows(block, col_count):
    if col_count is None or int(block.shape[0]) <= col_count:
        return None
    return fnp.mean(block[col_count:, :], axis=0)

def zero_diagonal_owned(matrix: fnp.ndarray) -> fnp.ndarray:
    fnp.fill_diagonal(matrix, 0.0)
    return matrix

def with_diagonal_owned(matrix: fnp.ndarray, diagonal: fnp.ndarray) -> fnp.ndarray:
    fnp.fill_diagonal(matrix, diagonal)
    return matrix

def symmetrize_matrix(matrix: fnp.ndarray) -> fnp.ndarray:
    return (matrix + matrix.T) * 0.5

def symmetrized_sum(matrix: fnp.ndarray) -> fnp.ndarray:
    return matrix + matrix.T

def history_rows(a, ids, cache=None):
    if ids is None:
        return a
    if isinstance(ids, tuple):
        if ids[0] == 0 and ids[1] >= int(a.shape[0]):
            return a
        if cache is not None:
            key = (id(a), ids)
            found = cache.get(key)
            if found is None:
                found = (a, a[ids[0]:ids[1], :])
                cache[key] = found
            return found[1]
        return a[ids[0]:ids[1], :]
    if cache is None:
        return a[ids, :]
    key = ('indexed_rows', id(a), id(ids))
    found = cache.get(key)
    if found is None:
        found = (a, ids, a[ids, :])
        cache[key] = found
    return found[2]

def restore_history_rows(a, ids, inverse, row_space):
    if ids is None:
        return a
    trailing = tuple(a.shape[1:])
    if isinstance(ids, tuple):
        (lo, hi) = ids
        parts = [] if lo == 0 else [fnp.zeros((lo,) + trailing, dtype=fnp.float32)]
        parts.append(a)
        if hi < row_space:
            parts.append(fnp.zeros((row_space - hi,) + trailing, dtype=fnp.float32))
        return fnp.concatenate(parts, axis=0) if len(parts) > 1 else a
    return pad_axis(a, 0, row_space - int(a.shape[0]), 0)[inverse]

def feature_sum(items):
    iterator = iter(items)
    result = next(iterator)
    for item in iterator:
        result = result + item
    return result

def elementwise_stack(parts, width, axis, cache=None):
    if cache is None:
        return fnp.stack([pad_axis(p, 0, max(0, width - int(p.shape[axis])), axis) for p in parts], axis=0)
    batch_key = ('stack', tuple(id(p) for p in parts), int(width), int(axis))
    found_batch = cache.get(batch_key)
    if found_batch is not None:
        return found_batch[1]
    padded = []
    for part in parts:
        after = max(0, width - int(part.shape[axis]))
        if after == 0:
            padded.append(part)
            continue
        key = (id(part), int(width), int(axis))
        found = cache.get(key)
        if found is None:
            found = (part, pad_axis(part, 0, after, axis))
            cache[key] = found
        padded.append(found[1])
    result = fnp.stack(padded, axis=0)
    cache[batch_key] = (tuple(parts), result)
    return result


def prefix_view(a, stop, axis=0):
    # A full prefix is the existing view, with identical storage and values.
    if stop is None or int(stop) >= int(a.shape[axis]):
        return a
    selection = [slice(None)] * len(a.shape)
    selection[axis] = slice(None, stop)
    return a[tuple(selection)]
