"""Training-only signatures for current owned-buffer kernels; retain autograd."""
import ast
import functools
import torch


def prepare_source(source):
    """Lower negative views and fused decode to identical ordered scalar sums."""
    if '_kz_' not in source:
        return source
    tree = ast.parse(source)
    matches = dict.fromkeys(('_kz_blocked', '_kz_blocked_top', '_kz_decode'), 0)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in matches:
            continue
        matches[node.name] += 1
        if node.name in ('_kz_blocked', '_kz_blocked_top'):
            assignments = [item for item in ast.walk(node) if isinstance(item, ast.Assign)
                           and isinstance(item.value, ast.Subscript)
                           and isinstance(item.value.value, ast.Name)
                           and item.value.value.id == 'split']
            if len(assignments) != 1:
                raise ValueError('negative-view lowering must match exactly once')
            axes = 'p + levels + 1' if node.name == '_kz_blocked' else 'bits'
            assignments[0].value = ast.parse(
                'fnp.flip(split, axis=tuple(range(' + axes + ', ' + axes + ' + levels)))',
                mode='eval').body
        else:
            branches = [item for item in ast.walk(node) if isinstance(item, ast.If)
                        and any(isinstance(part, ast.Name) and part.id == 'u2_c11'
                                for statement in item.body for part in ast.walk(statement))]
            if len(branches) != 1:
                raise ValueError('fused decode lowering must match exactly once')
            branches[0].body = ast.parse("""
deeper = _kz_regions(schemes, _KZ_OUT, level)
view = products[(slice(None),) * base + deeper] if deeper else products
axis = base + len(schemes) - 1 - level
q = fnp.unstack(view, axis=axis)
fnp.add(q[1], q[5], out=q[5])
fnp.add(q[1], q[3], out=q[3])
fnp.add(q[5], q[6], out=q[6])
fnp.add(q[5], q[0], out=q[1])
fnp.add(q[6], q[4], out=q[4])
fnp.add(q[1], q[2], out=q[5])
fnp.add(q[6], q[0], out=q[6])
continue
""").body
    if not any(matches.values()):
        # Earlier scientific sources contain no kz circuits. Keep their source
        # bytes intact rather than require this newer circuit family.
        return source
    if any(count != 1 for count in matches.values()):
        raise ValueError('current packed circuit lowering has unexpected source: ' + repr(matches))
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def prepare_backend(backend):
    """Keep the source's exact encode/product/decode order and saved layouts."""
    def live_view(value):
        # A destination view captured before its owned base first acquires a
        # grad_fn otherwise retains stale leaf metadata. Rebuild that alias
        # against the base's current autograd history at each ordered access.
        if isinstance(value, torch.Tensor) and value._base is not None:
            return value._base.as_strided(value.size(), value.stride(), value.storage_offset())
        return value

    def copyto(destination, source, casting=None):
        # torch.*(out=...) disables autograd; a copy into the source's owned
        # destination retains a CopySlices edge and the source's rounding order.
        live_view(destination).copy_(live_view(source))

    def asarray_chkfinite(value, dtype=None):
        result = backend.asarray(value, dtype=dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('array must not contain infs or NaNs')
        return result

    def std(value, axis=None, dtype=None, keepdims=False, ddof=0):
        if dtype is not None:
            value = value.to(dtype=dtype)
        return torch.std(value, dim=axis, keepdim=keepdims, correction=ddof)

    def polyval(coefficients, value):
        coefficients, value = backend.asarray(coefficients), backend.asarray(value)
        result = torch.zeros_like(value)
        for coefficient in coefficients:
            result = result * value + coefficient
        return result

    additions = dict(copyto=copyto, matmul=torch.matmul,
                     asarray_chkfinite=asarray_chkfinite, std=std, polyval=polyval,
                     logical_not=torch.logical_not, logical_or=torch.logical_or)
    for name, function in additions.items():
        if not hasattr(backend, name):
            setattr(backend, name, function)

    # unbind creates sibling outputs whose in-place modification is forbidden
    # once an owned destination acquires a grad_fn. Independent select views
    # retain the identical layout and permit the source's ordered buffer writes.
    backend.unstack = lambda value, axis=0: [value.select(axis, index) for index in range(value.shape[axis])]
    backend.flip = lambda value, axis=0: torch.flip(value, dims=axis if isinstance(axis, tuple) else (axis,))

    def owned_binary(original):
        @functools.wraps(original)
        def operation(left, right, out=None, dtype=None):
            left, right = live_view(left), live_view(right)
            if dtype is not None:
                left, right = backend.asarray(left, dtype), backend.asarray(right, dtype)
            # Without an owned output, call the exact existing arithmetic path.
            result = original(left, right)
            if out is not None:
                copyto(out, result)
                return out
            return result
        return operation

    defaults = dict(add=torch.add, subtract=torch.subtract, multiply=torch.multiply)
    for name, default in defaults.items():
        setattr(backend, name, owned_binary(getattr(backend, name, default)))
    return backend
