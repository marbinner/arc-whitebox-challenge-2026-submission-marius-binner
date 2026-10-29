"""General matrix multiplication with arithmetic recursion and storage rules (the v210 kernel).

Seven-product Winograd identities preserve every product term. The rule uses
only matrix dimensions; no fixture identifiers or exact-shape plans exist.
Owned padding retains all entries and is cropped after multiplication.

Families, all exact: the CK/KW/KV recursive dispatch; kz in-place packed levels
(Winograd 'w' and alternative-basis 'k' schemes, fused 'w' decode); rank-40
<3,3,6>/<3,6,3> tops (Dumas-Pernet-Sedoglavic circuits, CeCILL-B) with
per-predict caps; rank-48 tops (off). Per-predict state (_KZ_R40_TAKEN, the SL
unpack budget) is reset by Estimator._predict_impl. Knobs live in registry.py.
"""
from __future__ import annotations
from .config import KERNEL_CONFIG
from .state import fnp, math

def _winograd_sum(key, sums, cache, slots=None):
    if key in cache:
        return cache[key]
    (op, x, y) = sums[key]
    if isinstance(x, int):
        x = _winograd_sum(x, sums, cache, slots)
    if isinstance(y, int):
        y = _winograd_sum(y, sums, cache, slots)
    out = slots.get(key) if slots is not None else None
    value = fnp.add(x, y, out=out) if op == '+' else fnp.subtract(x, y, out=out)
    cache[key] = value
    return value
_WINOGRAD_OUTPUT = ((1, 1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1, 0), (1, 0, 0, -1, 0, 1, 1), (1, 0, 0, 0, 1, 1, 1))
_WINOGRAD_LEFT_GRID = (((1, 0), (0, 0)), ((0, 1), (0, 0)), ((1, 1), (-1, -1)), ((0, 0), (0, 1)), ((0, 0), (1, 1)), ((-1, 0), (1, 1)), ((1, 0), (-1, 0)))
_WINOGRAD_RIGHT_GRID = (((1, 0), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, -1), (-1, 1)), ((-1, 1), (0, 0)), ((1, -1), (0, 1)), ((0, -1), (0, 1)))
_WINOGRAD_MATRICES = [None, None, None]
_KV_OUTPUT_GRID = None

_KZ_R40_TAKEN = [0, 0]

def _kz_reset_routes():
    """Reset the per-predict count of capped rank-40 roots (in setup and at the start of every predict)."""
    _KZ_R40_TAKEN[:] = [0, 0]

def _setup_matmul():
    global _KV_OUTPUT_GRID
    _kz_reset_routes()
    _kz_sync_decode()
    _KV_OUTPUT_GRID = fnp.asarray(((_WINOGRAD_OUTPUT[0], _WINOGRAD_OUTPUT[1]), (_WINOGRAD_OUTPUT[2], _WINOGRAD_OUTPUT[3])), dtype=fnp.float32)
    _WINOGRAD_MATRICES[:] = [fnp.asarray(matrix, dtype=fnp.float32) for matrix in (_WINOGRAD_LEFT_GRID, _WINOGRAD_RIGHT_GRID, _WINOGRAD_OUTPUT)]
_WINOGRAD_GROUP_COUNTS = {(1, 0): ((1, 0, 1), (0, 0, 0)), (1, 1): ((1, 0, 1), (1, 0, 1)), (1, 2): ((1, 0, 1), (1, 0, 1)), (1, 3): ((0, 0, 0), (1, 0, 1)), (1, 4): ((0, 1, 0), (0, 1, 0)), (1, 5): ((0, 1, 0), (0, 1, 0)), (1, 6): ((0, 0, 0), (0, 0, 0)), (2, 0): ((2, 0, 1), (1, 0, 1)), (2, 2): ((1, 0, 1), (2, 0, 1)), (2, 4): ((0, 2, 0), (0, 2, 0)), (2, 6): ((0, 0, 0), (0, 0, 0)), (3, 0): ((3, 0, 1), (2, 0, 1)), (3, 3): ((0, 0, 0), (1, 2, 1)), (3, 6): ((0, 0, 0), (0, 0, 0)), (4, 0): ((3, 0, 1), (3, 0, 1)), (4, 4): ((0, 2, 0), (0, 2, 0)), (5, 0): ((3, 0, 1), (3, 0, 1)), (5, 5): ((0, 1, 0), (0, 1, 0)), (6, 0): ((3, 0, 1), (3, 0, 1)), (6, 6): ((0, 0, 0), (0, 0, 0)), (7, 0): ((3, 0, 1), (3, 0, 1))}

def _winograd_slab(plain, sums, keys, cache, operand, input_threshold, packing_counts):
    borrow_single = int(KERNEL_CONFIG.get('kw_borrow_single_slab', 0))
    if borrow_single > 0 and len(keys) == 1 and (keys[0] in plain or borrow_single >= 2):
        value = plain[keys[0]] if keys[0] in plain else _winograd_sum(keys[0], sums, cache)
        return value[..., None, :, :]
    shape = operand.shape[:-2] + (len(keys), operand.shape[-2] // 2, operand.shape[-1] // 2)
    (plain_count, prior_count, runs) = packing_counts
    new_sums = len(keys) - plain_count - prior_count
    use_stack = new_sums * (math.prod(shape) // len(keys)) <= -input_threshold * (1 + 2 * (runs + prior_count))
    if use_stack:
        return fnp.stack([plain[key] if key in plain else _winograd_sum(key, sums, cache) for key in keys], axis=-3)
    slab = fnp.empty(shape, dtype=operand.dtype)
    views = fnp.unstack(slab, axis=-3)
    slots = {key: views[i] for (i, key) in enumerate(keys) if key in sums}
    prior = set(cache)
    offset = 0
    while offset < len(keys):
        key = keys[offset]
        if key in plain:
            end = offset + 1
            while end < len(keys) and keys[end] in plain:
                end += 1
            if end == offset + 1 and int(KERNEL_CONFIG.get('kv_copy_singles', 0)):
                fnp.copyto(views[offset], plain[key])
            else:
                fnp.stack([plain[j] for j in keys[offset:end]], axis=-3, out=slab[..., offset:end, :, :])
            offset = end
        else:
            if key in prior:
                if int(KERNEL_CONFIG.get('kv_copy_singles', 0)):
                    fnp.copyto(views[offset], cache[key])
                else:
                    fnp.stack([cache[key]], axis=-3, out=slab[..., offset:offset + 1, :, :])
            elif key not in cache:
                _winograd_sum(key, sums, cache, slots)
            offset += 1
    return slab

def _quarters(matrix):
    """Return the four matrix blocks with the original split convention."""
    slice_limit = int(KERNEL_CONFIG.get('kw_slice_plane_limit', 0))
    rows, columns = matrix.shape[-2:]
    if int(KERNEL_CONFIG.get('kw_quarter_slices', 0)) and (slice_limit <= 0 or rows * columns <= slice_limit):
        m, n = rows // 2, columns // 2
        return (matrix[..., :m, :n], matrix[..., :m, n:],
                matrix[..., m:, :n], matrix[..., m:, n:])
    (top, bottom) = fnp.split(matrix, 2, axis=-2)
    (a11, a12) = fnp.split(top, 2, axis=-1)
    (a21, a22) = fnp.split(bottom, 2, axis=-1)
    return (a11, a12, a21, a22)

def _matmul_operand(parts):
    """Borrow an identical shared basis and broadcast it over products."""
    first = parts[0]
    if all(part is first for part in parts):
        return first[None, ...]
    return fnp.stack(parts, axis=0)

def _matmul_batch_size(left, right):
    """Count output batch planes from cached shape metadata, including broadcasts."""
    left_batch, right_batch = left.shape[:-2], right.shape[:-2]
    if left_batch == right_batch:
        return math.prod(left_batch)
    size = 1
    for axis in range(1, max(len(left_batch), len(right_batch)) + 1):
        a = left_batch[-axis] if axis <= len(left_batch) else 1
        b = right_batch[-axis] if axis <= len(right_batch) else 1
        if a != b and a != 1 and b != 1:
            raise ValueError('Incompatible matrix batch dimensions')
        size *= b if a == 1 else a
    return size

def _sparse_leaf_encode(matrix, is_left, levels):
    value = matrix
    for stage in range(levels):
        a, b, c, d = _quarters(value)
        if int(KERNEL_CONFIG.get('kw_sparse_leaf', 0)) >= 3:
            if is_left:
                plain = {0: a, 1: b, 3: d}
                sums = {4: ('+', c, d), 5: ('-', 4, a),
                        6: ('-', a, c), 2: ('-', b, 5)}
                counts = (3, 0, 2)
            else:
                plain = {0: a, 1: c, 2: d}
                sums = {4: ('-', b, a), 5: ('-', d, 4),
                        6: ('-', d, b), 3: ('-', 5, c)}
                counts = (3, 0, 1)
            value = _winograd_slab(plain, sums, tuple(range(7)), {}, value, 0, counts)
            continue
        if is_left:
            s1 = fnp.add(c, d)
            s2 = fnp.subtract(s1, a)
            s3 = fnp.subtract(a, c)
            s4 = fnp.subtract(b, s2)
            parts = (a, b, s4, d, s1, s2, s3)
        else:
            t1 = fnp.subtract(b, a)
            t2 = fnp.subtract(d, t1)
            t3 = fnp.subtract(d, b)
            t4 = fnp.subtract(t2, c)
            parts = (a, c, d, t4, t1, t2, t3)
        value = fnp.stack(parts, axis=-3)
    return value

def _separable_leaf(left, right, levels, request_price):
    """Compile consecutive binary encodings along owned block coordinates."""
    decode64 = bool(int(KERNEL_CONFIG.get('kw_leaf_decode64', 0)))
    if decode64 or int(KERNEL_CONFIG.get('kw_pairwise_decode', 0)) == 1:
        request_price = 0
    encoded = []
    (ranks, row_bits, column_bits) = ('pqr', 'abc', 'def')
    for (matrix, coefficients) in zip((left, right), _WINOGRAD_MATRICES[:2]):
        dense_limit = int(KERNEL_CONFIG.get('kv_dense_encode_elements', 0))
        use_sparse = int(KERNEL_CONFIG.get('kw_sparse_leaf', 0)) and not (
            dense_limit > 0 and math.prod(matrix.shape) // (4 ** levels) <= dense_limit)
        if use_sparse:
            encoded.append(_sparse_leaf_encode(matrix, len(encoded) == 0, levels))
            continue
        (m, k) = matrix.shape[-2:]
        shape = matrix.shape[:-2] + (2,) * levels + (m // 2 ** levels,) + (2,) * levels + (k // 2 ** levels,)
        value = fnp.reshape(matrix, shape, copy=True)
        labels = row_bits[:levels] + 'i' + column_bits[:levels] + 'j'
        for stage in range(levels):
            result = ranks[:stage + 1] + row_bits[stage + 1:levels] + 'i' + column_bits[stage + 1:levels] + 'j'
            expression = ranks[stage] + row_bits[stage] + column_bits[stage] + ',...' + labels + '->...' + result
            value = fnp.einsum(expression, coefficients, value, optimize=False)
            labels = result
        encoded.append(value)
    if int(KERNEL_CONFIG.get('kw_sparse_leaf', 0)) in (2, 4):
        request_price = 0
    unit = _matmul_batch_size(encoded[0], encoded[1]) // 7 * encoded[0].shape[-2] * encoded[1].shape[-1]
    if 45 * unit <= 7 * request_price:
        if int(KERNEL_CONFIG.get('kv_matrix_decode', 0)):
            value = fnp.einsum('...rij,...rjk,abr->...aibk', encoded[0], encoded[1], _KV_OUTPUT_GRID, optimize=[(0, 1), (0, 1)])
            value = fnp.reshape(value, value.shape[:-4]+(value.shape[-4]*value.shape[-3], value.shape[-2]*value.shape[-1]), copy=True)
        else:
            value = fnp.einsum('...rij,...rjk,or->...oik', encoded[0], encoded[1], _WINOGRAD_MATRICES[2], optimize=[(0, 1), (0, 1)])
            (a, b, c, d) = fnp.unstack(value, axis=-3)
            value = fnp.block([[a, b], [c, d]])
        remaining = levels - 1
    else:
        value = encoded[0] @ encoded[1]
        remaining = levels
    if decode64:
        original_dtype = value.dtype
        value = value.astype(fnp.float64)
    for stage in range(remaining):
        unit = math.prod(value.shape) // 7
        if 45 * unit <= 6 * request_price:
            if int(KERNEL_CONFIG.get('kv_matrix_decode', 0)):
                value = fnp.einsum('abr,...rij->...aibj', _KV_OUTPUT_GRID, value, optimize=False)
                value = fnp.reshape(value, value.shape[:-4]+(value.shape[-4]*value.shape[-3], value.shape[-2]*value.shape[-1]), copy=True)
                continue
            value = fnp.einsum('or,...rij->...oij', _WINOGRAD_MATRICES[2], value, optimize=False)
            (a, b, c, d) = fnp.unstack(value, axis=-3)
        else:
            (p1, p2, p3, p4, p5, p6, p7) = fnp.unstack(value, axis=-3)
            if int(KERNEL_CONFIG.get('kw_pairwise_decode', 0)) == 1:
                a = fnp.add(p1,p2)
                common = fnp.add(p1,p6)
                b = fnp.add(common,fnp.add(p3,p5))
                c = fnp.add(common,fnp.subtract(p7,p4))
                d = fnp.add(common,fnp.add(p5,p7))
                value = fnp.block([[a,b],[c,d]])
                continue
            a = fnp.add(p1, p2)
            u2 = fnp.add(p1, p6)
            u3 = fnp.add(u2, p7)
            d = fnp.add(u3, p5)
            u4 = fnp.add(u2, p5)
            b = fnp.add(u4, p3)
            c = fnp.subtract(u3, p4)
        value = fnp.block([[a, b], [c, d]])
    if decode64:
        value = value.astype(original_dtype)
    return value

def _winograd_matmul(left, right, levels, threshold, schedule):
    if levels <= 0:
        return left @ right
    (caps, leaf, input_threshold, output_threshold, separable_threshold, decoder_price) = schedule
    (rows, inner) = left.shape[-2:]
    columns = right.shape[-1]
    if min(rows, inner, columns) <= threshold or rows % 2 or inner % 2 or columns % 2:
        return left @ right
    three_level_limit = int(KERNEL_CONFIG.get('kw_three_leaf_elements', 0))
    if levels == 3 and three_level_limit > 0 and not (rows % 8 or inner % 8 or columns % 8):
        three_elements = max(math.prod(left.shape),math.prod(right.shape),_matmul_batch_size(left,right)*rows*columns)//64
        if three_elements <= three_level_limit:
            return _separable_leaf(left,right,levels,decoder_price)
    one_level_limit = int(KERNEL_CONFIG.get('kw_one_level_elements', 0))
    if levels == 1 and one_level_limit > 0:
        one_elements = max(math.prod(left.shape), math.prod(right.shape), _matmul_batch_size(left, right) * rows * columns) // 4
        if one_elements <= one_level_limit:
            return _separable_leaf(left, right, levels, decoder_price)
    if separable_threshold > 0 and (levels == 2 and (not (rows % 4 or inner % 4 or columns % 4))):
        batch = _matmul_batch_size(left, right)
        elements = max(math.prod(left.shape), math.prod(right.shape), batch * rows * columns) // 16
        if elements <= separable_threshold:
            return _separable_leaf(left, right, levels, decoder_price)
    (a11, a12, a21, a22) = _quarters(left)
    (b11, b12, b21, b22) = _quarters(right)
    (m, k, n) = (rows // 2, inner // 2, columns // 2)
    batch = _matmul_batch_size(left, right)
    elements = batch * max(m * k, k * n, m * n)
    cap = caps[min(levels, 6) - 1]
    chunk = max(1, min(7, cap // max(elements, 1)))
    leaf = int(leaf)
    if levels <= leaf:
        chunk = 1
        pack_limit = int(KERNEL_CONFIG.get('kw_leaf_pack_elements', 0))
        if levels == 1 and pack_limit > 0 and elements <= pack_limit:
            chunk = max(1, min(7, cap // max(elements, 1)))
    lp = {0: a11, 1: a12, 3: a22}
    ls = {4: ('+', a21, a22), 5: ('-', 4, a11), 6: ('-', a11, a21), 2: ('-', a12, 5)}
    rp = {0: b11, 1: b21, 2: b22}
    rs = {4: ('-', b12, b11), 5: ('-', b22, 4), 6: ('-', b22, b12), 3: ('-', 5, b21)}
    (lc, rc) = ({}, {})
    products = [None] * 7
    order = (3, 0, 1, 2, 4, 5, 6)
    for start in range(0, 7, chunk):
        keys = order[start:start + chunk]
        if len(keys) == 1:
            key = keys[0]
            aa = lp[key] if key in lp else _winograd_sum(key, ls, lc)
            bb = rp[key] if key in rp else _winograd_sum(key, rs, rc)
            products[key] = _winograd_matmul(aa, bb, levels - 1, threshold, schedule)
        else:
            packing_counts = _WINOGRAD_GROUP_COUNTS[chunk, start]
            aa = _winograd_slab(lp, ls, keys, lc, left, input_threshold, packing_counts[0])
            bb = _winograd_slab(rp, rs, keys, rc, right, input_threshold, packing_counts[1])
            pp = _winograd_matmul(aa, bb, levels - 1, threshold, schedule)
            for (j, p) in zip(keys, fnp.unstack(pp, axis=-3)):
                products[j] = p
    output_dtype = products[0].dtype
    wide_output_minimum = int(KERNEL_CONFIG.get('kw_output_decode64', 0))
    wide_output = wide_output_minimum > 0 and levels >= wide_output_minimum
    if wide_output:
        products = [value.astype(fnp.float64) for value in products]
    (p1, p2, p3, p4, p5, p6, p7) = products
    pairwise_mode = int(KERNEL_CONFIG.get('kw_pairwise_decode', 0))
    if pairwise_mode in (1,2) or (pairwise_mode == 3 and levels >= 4):
        if math.prod(p1.shape[:-2]) * rows * columns <= int(output_threshold):
            a = fnp.add(p1,p2,out=p2)
            common = fnp.add(p1,p6,out=p6)
            bp = fnp.add(p3,p5,out=p3)
            b = fnp.add(common,bp,out=p3)
            cp = fnp.subtract(p7,p4,out=p4)
            c = fnp.add(common,cp,out=p4)
            dp = fnp.add(p5,p7,out=p5)
            d = fnp.add(common,dp,out=p1)
            result = fnp.block([[a,b],[c,d]])
        else:
            result = fnp.empty(p1.shape[:-2] + (rows,columns),dtype=p1.dtype)
            a,b,c,d = _quarters(result)
            fnp.add(p1,p2,out=a)
            common = fnp.add(p1,p6,out=p6)
            fnp.add(p3,p5,out=p3)
            fnp.add(common,p3,out=b)
            fnp.subtract(p7,p4,out=p4)
            fnp.add(common,p4,out=c)
            fnp.add(p5,p7,out=p5)
            fnp.add(common,p5,out=d)
        return result.astype(output_dtype) if wide_output else result
    if math.prod(p1.shape[:-2]) * rows * columns <= int(output_threshold):
        u1 = fnp.add(p1, p2, out=p2)
        u2 = fnp.add(p1, p6, out=p6)
        u3 = fnp.add(u2, p7, out=p7)
        u7 = fnp.add(u3, p5, out=p1)
        u4 = fnp.add(u2, p5, out=p5)
        u5 = fnp.add(u4, p3, out=p3)
        u6 = fnp.subtract(u3, p4, out=p4)
        result = fnp.block([[u1, u5], [u6, u7]])
        return result.astype(output_dtype) if wide_output else result
    result = fnp.empty(p1.shape[:-2] + (rows, columns), dtype=p1.dtype)
    (c11, c12, c21, c22) = _quarters(result)
    fnp.add(p1, p2, out=c11)
    u2 = fnp.add(p1, p6, out=p6)
    u3 = fnp.add(u2, p7, out=p7)
    fnp.add(u3, p5, out=c22)
    u4 = fnp.add(u2, p5, out=p5)
    fnp.add(u4, p3, out=c12)
    fnp.subtract(u3, p4, out=c21)
    return result.astype(output_dtype) if wide_output else result

"""Source fragment: classical seven-product Strassen, sparse packed leaves."""

def _kw_classical_parts(matrix, is_left):
    a, b, c, d = _quarters(matrix)
    if is_left:
        return (fnp.add(a,d), fnp.add(c,d), a, d,
                fnp.add(a,b), fnp.subtract(c,a), fnp.subtract(b,d))
    return (fnp.add(a,d), a, fnp.subtract(b,d), fnp.subtract(c,a),
            d, fnp.add(a,b), fnp.add(c,d))

def _kw_classical_decode(parts):
    p1,p2,p3,p4,p5,p6,p7 = parts
    if int(KERNEL_CONFIG.get('kw_balanced_classical_decode', 0)):
        a = fnp.add(fnp.add(p1,p4),fnp.subtract(p7,p5))
        b = fnp.add(p3,p5)
        c = fnp.add(p2,p4)
        d = fnp.add(fnp.subtract(p1,p2),fnp.add(p3,p6))
        return fnp.block([[a,b],[c,d]])
    a = fnp.add(fnp.subtract(fnp.add(p1,p4),p5),p7)
    b = fnp.add(p3,p5)
    c = fnp.add(p2,p4)
    d = fnp.add(fnp.add(fnp.subtract(p1,p2),p3),p6)
    return fnp.block([[a,b],[c,d]])

def _kw_classical_leaf(left, right, levels):
    encoded=[]
    for matrix,is_left in ((left,True),(right,False)):
        for stage in range(levels):
            matrix=fnp.stack(_kw_classical_parts(matrix,is_left),axis=-3)
        encoded.append(matrix)
    result=encoded[0] @ encoded[1]
    for stage in range(levels):
        result=_kw_classical_decode(fnp.unstack(result,axis=-3))
    return result

def _kw_classical_packing(keys, plain):
    count=sum(key in plain for key in keys)
    runs=sum(key in plain and (i==0 or keys[i-1] not in plain) for i,key in enumerate(keys))
    return (count,0,runs)

def _kw_classical_matmul(left,right,levels,threshold,schedule,classical_levels=None):
    if classical_levels is None:
        prefix = int(KERNEL_CONFIG.get('kw_classical_prefix', 0))
        classical_levels = min(prefix, levels) if prefix > 0 else levels
    if classical_levels <= 0:
        return _winograd_matmul(left,right,levels,threshold,schedule)
    if levels<=0:
        return left @ right
    caps,leaf,input_threshold,output_threshold,separable_threshold,decoder_price=schedule
    rows,inner=left.shape[-2:];columns=right.shape[-1]
    if min(rows,inner,columns)<=threshold or rows%2 or inner%2 or columns%2:
        return left @ right
    batch=_matmul_batch_size(left,right)
    if levels==2 and classical_levels>=levels and separable_threshold>0 and not (rows%4 or inner%4 or columns%4):
        elements=max(math.prod(left.shape),math.prod(right.shape),batch*rows*columns)//16
        if elements<=separable_threshold:
            return _kw_classical_leaf(left,right,levels)
    a,b,c,d=_quarters(left);e,f,g,h=_quarters(right)
    lp={2:a,3:d}
    ls={0:('+',a,d),1:('+',c,d),4:('+',a,b),5:('-',c,a),6:('-',b,d)}
    rp={1:e,4:h}
    rs={0:('+',e,h),2:('-',f,h),3:('-',g,e),5:('+',e,f),6:('+',g,h)}
    lc,rc={},{}
    elements=batch*max(rows//2*inner//2,inner//2*columns//2,rows//2*columns//2)
    cap=caps[min(levels,6)-1]
    chunk=max(1,min(7,cap//max(elements,1)))
    if levels<=int(leaf):chunk=1
    products=[None]*7
    order=(3,0,1,2,4,5,6)
    for start in range(0,7,chunk):
        keys=order[start:start+chunk]
        if len(keys)==1:
            key=keys[0]
            aa=lp[key] if key in lp else _winograd_sum(key,ls,lc)
            bb=rp[key] if key in rp else _winograd_sum(key,rs,rc)
            products[key]=_kw_classical_matmul(aa,bb,levels-1,threshold,schedule,classical_levels-1)
        else:
            aa=_winograd_slab(lp,ls,keys,lc,left,input_threshold,_kw_classical_packing(keys,lp))
            bb=_winograd_slab(rp,rs,keys,rc,right,input_threshold,_kw_classical_packing(keys,rp))
            value=_kw_classical_matmul(aa,bb,levels-1,threshold,schedule,classical_levels-1)
            for key,part in zip(keys,fnp.unstack(value,axis=-3)):
                products[key]=part
    # Decode in the same order as the packed leaf. Every product term survives.
    return _kw_classical_decode(products)


def _kernel_schedule():
    return ((int(KERNEL_CONFIG['ck_workspace_elements']),) * 6, int(KERNEL_CONFIG['ck_unpacked_levels']), int(KERNEL_CONFIG['ck_input_threshold']), (0 if int(KERNEL_CONFIG.get('kw_owned_decode', 0)) else KERNEL_CONFIG['block_output_max_elements']), int(KERNEL_CONFIG['ck_separable_elements']), int(KERNEL_CONFIG['ck_decoder_price']))

def _kw_forward_matmul(left, right, leg_transport=False):
    (m, k) = left.shape[-2:]
    n = right.shape[-1]
    batch = _matmul_batch_size(left, right)
    if batch == 0:
        return left @ right
    leaf = int(KERNEL_CONFIG['ck_leaf_size'])
    large_volume = int(KERNEL_CONFIG.get('kw_large_volume', 0))
    below_minimum = min(m, k, n) < int(KERNEL_CONFIG['ck_min_dimension'])
    large_divisor = int(KERNEL_CONFIG.get('kw_large_min_divisor', 0)) or 2
    large_half_minimum = large_volume > 0 and min(m, k, n) >= int(KERNEL_CONFIG['ck_min_dimension']) // large_divisor and (batch * m * k * n >= large_volume)
    plan_minimum = min(m, k, n)
    near_distance = int(KERNEL_CONFIG.get('kw_near_min_padding', 0))
    near_padding = False
    if below_minimum and near_distance > 0 and (int(KERNEL_CONFIG['ck_pad_multiple']) > 0):
        near_multiple = int(KERNEL_CONFIG['ck_pad_multiple'])
        (qm, qk, qn) = ((v + near_multiple - 1) // near_multiple * near_multiple for v in (m, k, n))
        near_padding = plan_minimum >= int(KERNEL_CONFIG['ck_min_dimension']) - near_distance and min(qm, qk, qn) >= int(KERNEL_CONFIG['ck_min_dimension']) and (qm * qk * qn * 10000 <= m * k * n * int(KERNEL_CONFIG['ck_pad_limit']))
        if near_padding:
            plan_minimum = min(qm, qk, qn)
    if below_minimum and (not (large_half_minimum or near_padding)):
        return left @ right
    depth = min(int(KERNEL_CONFIG['ck_max_depth']), max(0, (plan_minimum // leaf).bit_length() - 1))
    if int(KERNEL_CONFIG['ck_thin_depth']) > 0 and n * int(KERNEL_CONFIG['ck_aspect_ratio']) <= min(m, k) and (batch == 1):
        thin_cap = int(KERNEL_CONFIG['ck_thin_depth'])
        wide_ratio = int(KERNEL_CONFIG.get('kw_thin_wide_ratio', 0))
        wide_depth = int(KERNEL_CONFIG.get('kw_thin_wide_depth', 0))
        if wide_ratio > 0 and wide_depth > 0 and (k >= m * wide_ratio):
            thin_cap = wide_depth
        depth = min(depth, thin_cap)
    if int(KERNEL_CONFIG['ck_cov_depth']) > 0 and m == n and (k >= m * int(KERNEL_CONFIG['ck_aspect_ratio'])) and (m <= int(KERNEL_CONFIG['ck_cov_plane_limit'])):
        depth = min(depth, int(KERNEL_CONFIG['ck_cov_depth']))
    shallow_volume = int(KERNEL_CONFIG.get('kw_shallow_volume', 0))
    if shallow_volume > 0 and depth >= 6 and (batch * m * k * n <= shallow_volume):
        depth = 5
    if int(KERNEL_CONFIG.get('kw_deep_wide', 0)) and batch == 1 and (m == k) and (m >= 1024) and (n >= 4 * m):
        depth = max(depth, 6 + int(KERNEL_CONFIG['kw_deep_wide']))
    multiple = int(KERNEL_CONFIG['ck_pad_multiple'])
    covariance_rule = int(KERNEL_CONFIG.get('kw_cov_dispatch', 0))
    if covariance_rule > 0 and int(KERNEL_CONFIG['ck_cov_depth']) > 0 and (m == n) and (m <= int(KERNEL_CONFIG['ck_cov_plane_limit'])):
        covariance_cap = covariance_rule == 2 and k >= m
        if covariance_rule == 1 and multiple > 0:
            (cm, ck, cn) = ((v + multiple - 1) // multiple * multiple for v in (m, k, n))
            covariance_cap = ck >= cm * int(KERNEL_CONFIG['ck_aspect_ratio']) and cm <= int(KERNEL_CONFIG['ck_cov_plane_limit']) and (cm * ck * cn * 10000 <= m * k * n * int(KERNEL_CONFIG['ck_pad_limit']))
        if covariance_cap:
            depth = min(depth, int(KERNEL_CONFIG['ck_cov_depth']))
    tiny_short = int(KERNEL_CONFIG.get('kw_tiny_leaf_short', 0))
    tiny_long = int(KERNEL_CONFIG.get('kw_tiny_leaf_long', 0))
    product_limit = int(KERNEL_CONFIG.get('kw_leaf_product_limit', 0))
    tiny_work = int(KERNEL_CONFIG.get('kz_tiny_work', 0))
    step = 1 << depth
    padded_work = batch * (-(-m // step) * step) * (-(-k // step) * step) * (-(-n // step) * step)
    tiny_exempt = tiny_work > 0 and batch * m * k * n >= tiny_work and (depth >= int(KERNEL_CONFIG.get('kv_packed_depth_min', 0))) and (padded_work == batch * m * k * n or (int(KERNEL_CONFIG.get('kz_tiny_pad', 0)) > 0 and padded_work * 10000 <= batch * m * k * n * int(KERNEL_CONFIG['ck_pad_limit'])))
    if tiny_short > 0 and tiny_long > 0 and (not tiny_exempt):
        while depth > 0:
            divisor = 2 ** depth
            leaf_dimensions = sorted(((v + divisor - 1) // divisor for v in (m, k, n)))
            if leaf_dimensions[1] > tiny_short or leaf_dimensions[2] > tiny_long:
                break
            depth -= 1
    if product_limit > 0:
        while depth > 0 and batch * 7 ** depth > product_limit:
            depth -= 1
    deep_work = int(KERNEL_CONFIG.get('kz_deep_work', 0))
    if deep_work > 0 and depth == int(KERNEL_CONFIG['ck_max_depth']) and (batch * m * k * n >= deep_work) and (not any((v % (2 << depth) for v in (m, k, n)))) and (min(m, k, n) >= leaf - 1 << depth + 1):
        depth += 1
    classical_plan = int(KERNEL_CONFIG.get('kw_classical_plan', 0))
    classical_root = classical_plan == 1 or (classical_plan == 2 and batch == 1 and (m == k) and (m >= 1024) and (n >= 4 * m))
    recursive_matmul = _kw_classical_matmul if classical_root else _winograd_matmul
    packed = bool(int(KERNEL_CONFIG.get('kv_packed_output', 0))) and (not classical_root)
    if packed:
        recursive_matmul = _kv_packed_root
    if _kv_r48_eligible(m, k, n, batch, depth):
        recursive_matmul = _kv_r48_packed_root if packed else _kv_r48_matmul
    padding_rule = int(KERNEL_CONFIG.get('kw_padding_policy', 0))
    if multiple > 0 and depth > 0 and (padding_rule > 0):
        multiple = min(multiple, 2 ** depth) if padding_rule == 1 else 2 ** depth
    schedule = _kernel_schedule()
    if leg_transport:
        depth, schedule = _leg_transport_schedule(m, k, n, batch, depth, leaf, multiple, schedule)
    if multiple > 0 and depth > 0:
        (pm, pk, pn) = ((v + multiple - 1) // multiple * multiple for v in (m, k, n))
        if pm * pk * pn * 10000 <= m * k * n * int(KERNEL_CONFIG['ck_pad_limit']):
            left = fnp.pad(left, [(0, 0)] * (len(left.shape) - 2) + [(0, pm - m), (0, pk - k)], mode='constant') if pm != m or pk != k else left
            right = fnp.pad(right, [(0, 0)] * (len(right.shape) - 2) + [(0, pk - k), (0, pn - n)], mode='constant') if pk != k or pn != n else right
            result = recursive_matmul(left, right, depth, leaf - 1, schedule)
            if pm == m and pn == n:
                return result
            return result[..., :m, :n]
    return recursive_matmul(left, right, depth, leaf - 1, schedule)

def _kw_matrix_transpose(matrix):
    axes = list(range(len(matrix.shape)))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return fnp.transpose(matrix, tuple(axes))

def _k3_fast_matmul(left, right, leg_transport=False):
    orientation = int(KERNEL_CONFIG.get('kw_product_orientation', 0))
    if orientation:
        (m, k) = left.shape[-2:]
        n = right.shape[-1]
        transpose = False
        if orientation in (1, 2):
            transpose = m * int(KERNEL_CONFIG['ck_aspect_ratio']) <= min(k, n) and (orientation == 2 or _matmul_batch_size(left, right) == 1)
        elif orientation == 3:
            transpose = math.prod(right.shape) > math.prod(left.shape)
        if transpose:
            result = _kw_forward_matmul(_kw_matrix_transpose(right), _kw_matrix_transpose(left), leg_transport)
            return _kw_matrix_transpose(result)
    return _kw_forward_matmul(left, right, leg_transport)


# PLinOpt rank-48 circuits: see NOTICE.txt and LICENSE_CeCILL-B.txt.
def _k48_r48_l_out(inputs, outputs, xp):
    """4x4x4_48_204_L.slp: 55 adds/subtracts + 2 scalings + 0 negations."""
    i0 = inputs[0]
    i1 = inputs[1]
    i2 = inputs[2]
    i3 = inputs[3]
    i4 = inputs[4]
    i5 = inputs[5]
    i6 = inputs[6]
    i7 = inputs[7]
    i8 = inputs[8]
    i9 = inputs[9]
    i10 = inputs[10]
    i11 = inputs[11]
    i12 = inputs[12]
    i13 = inputs[13]
    i14 = inputs[14]
    i15 = inputs[15]
    t10 = xp.add(i0, i3)
    t11 = xp.add(i1, i6)
    t14 = xp.add(i4, i7)
    t21 = xp.add(i8, i11)
    t25 = xp.add(i1, i13)
    t26 = xp.subtract(i6, i14)
    o0 = xp.add(i1, t10, out=outputs[0])
    o28 = xp.subtract(t14, i6, out=outputs[28])
    o2 = xp.subtract(i10, t21, out=outputs[2])
    o42 = xp.add(i9, t21, out=outputs[42])
    o36 = xp.subtract(t25, i5, out=outputs[36])
    o45 = xp.subtract(t25, i9, out=outputs[45])
    o22 = xp.subtract(t26, i2, out=outputs[22])
    o4 = xp.subtract(i10, t26, out=outputs[4])
    t12 = xp.true_divide(xp.add(t10, t14), 2)
    t13 = xp.subtract(o4, o45)
    o27 = xp.add(i1, t12, out=outputs[27])
    o21 = xp.subtract(t12, i8, out=outputs[21])
    o8 = xp.subtract(t12, i6, out=outputs[8])
    t15 = xp.subtract(t10, t12)
    o11 = xp.add(i12, t15, out=outputs[11])
    o43 = xp.true_divide(xp.add(t11, t13), 2, out=outputs[43])
    t16 = xp.add(i15, o11)
    o37 = xp.subtract(t11, o43, out=outputs[37])
    t17 = xp.subtract(o43, i14)
    t18 = xp.add(i13, o43)
    t19 = xp.add(o8, o43)
    t20 = xp.subtract(o37, o36)
    t22 = xp.subtract(o37, o22)
    o14 = xp.subtract(t17, o2, out=outputs[14])
    o39 = xp.subtract(i7, t20, out=outputs[39])
    o33 = xp.subtract(i3, t22, out=outputs[33])
    o24 = xp.subtract(t19, o14, out=outputs[24])
    o1 = xp.subtract(t18, o39, out=outputs[1])
    t23 = xp.add(o11, o39)
    o20 = xp.subtract(t17, o33, out=outputs[20])
    t24 = xp.add(o39, o33)
    o34 = xp.subtract(o24, t13, out=outputs[34])
    o32 = xp.subtract(t16, o24, out=outputs[32])
    o16 = xp.subtract(t23, o21, out=outputs[16])
    o18 = xp.subtract(t23, o33, out=outputs[18])
    o10 = xp.subtract(i11, t24, out=outputs[10])
    o31 = xp.subtract(t24, o21, out=outputs[31])
    o44 = xp.add(t16, o34, out=outputs[44])
    o47 = xp.add(t19, o34, out=outputs[47])
    o26 = xp.add(t20, o16, out=outputs[26])
    o46 = xp.add(t18, o16, out=outputs[46])
    o17 = xp.add(t15, o18, out=outputs[17])
    o5 = xp.subtract(t16, o18, out=outputs[5])
    o3 = xp.subtract(t23, o31, out=outputs[3])
    o19 = xp.subtract(t12, o31, out=outputs[19])
    o38 = xp.add(t20, o47, out=outputs[38])
    o6 = xp.subtract(t22, o47, out=outputs[6])
    o29 = xp.subtract(t22, o3, out=outputs[29])
    o13 = xp.subtract(t17, o3, out=outputs[13])
    xp.copyto(outputs[7], i11)
    o7 = outputs[7]
    xp.copyto(outputs[9], i13)
    o9 = outputs[9]
    xp.copyto(outputs[12], i14)
    o12 = outputs[12]
    xp.copyto(outputs[15], i15)
    o15 = outputs[15]
    xp.copyto(outputs[23], i6)
    o23 = outputs[23]
    xp.copyto(outputs[25], i1)
    o25 = outputs[25]
    xp.copyto(outputs[30], i7)
    o30 = outputs[30]
    xp.copyto(outputs[35], i8)
    o35 = outputs[35]
    xp.copyto(outputs[40], i12)
    o40 = outputs[40]
    xp.copyto(outputs[41], i3)
    o41 = outputs[41]
    return outputs

def _k48_r48_r_out(inputs, outputs, xp):
    """4x4x4_48_204_R.slp: 55 adds/subtracts + 2 scalings + 0 negations."""
    i0 = inputs[0]
    i1 = inputs[1]
    i2 = inputs[2]
    i3 = inputs[3]
    i4 = inputs[4]
    i5 = inputs[5]
    i6 = inputs[6]
    i7 = inputs[7]
    i8 = inputs[8]
    i9 = inputs[9]
    i10 = inputs[10]
    i11 = inputs[11]
    i12 = inputs[12]
    i13 = inputs[13]
    i14 = inputs[14]
    i15 = inputs[15]
    t10 = xp.subtract(i2, i14)
    t11 = xp.subtract(i6, i8)
    t14 = xp.subtract(i0, i12)
    t21 = xp.subtract(i1, i13)
    t25 = xp.add(i6, i7)
    t26 = xp.subtract(i8, i11)
    o26 = xp.subtract(i6, t10, out=outputs[26])
    o29 = xp.subtract(t14, i8, out=outputs[29])
    o22 = xp.subtract(t21, i9, out=outputs[22])
    o36 = xp.subtract(i5, t21, out=outputs[36])
    o42 = xp.subtract(i4, t25, out=outputs[42])
    o1 = xp.add(i5, t25, out=outputs[1])
    o2 = xp.subtract(i10, t26, out=outputs[2])
    o20 = xp.add(i9, t26, out=outputs[20])
    t12 = xp.true_divide(xp.add(t10, t14), 2)
    t13 = xp.subtract(o1, o20)
    o16 = xp.subtract(t12, i6, out=outputs[16])
    o18 = xp.add(i1, t12, out=outputs[18])
    o3 = xp.subtract(t12, i8, out=outputs[3])
    t15 = xp.subtract(t12, t10)
    o11 = xp.subtract(i3, t15, out=outputs[11])
    o43 = xp.true_divide(xp.subtract(t13, t11), 2, out=outputs[43])
    t16 = xp.subtract(o11, i15)
    o14 = xp.add(t11, o43, out=outputs[14])
    t17 = xp.subtract(o43, i11)
    t18 = xp.subtract(o43, i7)
    t19 = xp.add(o3, o43)
    t20 = xp.add(o42, o14)
    t22 = xp.subtract(o2, o14)
    o37 = xp.subtract(t17, o22, out=outputs[37])
    o34 = xp.subtract(t20, i12, out=outputs[34])
    o24 = xp.add(i14, t22, out=outputs[24])
    o33 = xp.subtract(t19, o37, out=outputs[33])
    o45 = xp.subtract(t18, o34, out=outputs[45])
    t23 = xp.add(o11, o34)
    o4 = xp.add(o24, t17, out=outputs[4])
    t24 = xp.subtract(o24, o34)
    o39 = xp.subtract(t13, o33, out=outputs[39])
    o41 = xp.subtract(t16, o33, out=outputs[41])
    o27 = xp.subtract(o18, t23, out=outputs[27])
    o21 = xp.add(o24, t23, out=outputs[21])
    o10 = xp.add(i13, t24, out=outputs[10])
    o31 = xp.add(o18, t24, out=outputs[31])
    o30 = xp.subtract(o39, t16, out=outputs[30])
    o47 = xp.subtract(t19, o39, out=outputs[47])
    o0 = xp.add(t20, o27, out=outputs[0])
    o25 = xp.subtract(o27, t18, out=outputs[25])
    o35 = xp.subtract(t15, o21, out=outputs[35])
    o7 = xp.subtract(o21, t16, out=outputs[7])
    o8 = xp.add(t23, o31, out=outputs[8])
    o19 = xp.add(t12, o31, out=outputs[19])
    o38 = xp.subtract(t20, o47, out=outputs[38])
    o6 = xp.subtract(t22, o47, out=outputs[6])
    o28 = xp.subtract(o8, t22, out=outputs[28])
    o23 = xp.subtract(o8, t17, out=outputs[23])
    xp.copyto(outputs[5], i13)
    o5 = outputs[5]
    xp.copyto(outputs[9], i7)
    o9 = outputs[9]
    xp.copyto(outputs[12], i11)
    o12 = outputs[12]
    xp.copyto(outputs[13], i8)
    o13 = outputs[13]
    xp.copyto(outputs[15], i15)
    o15 = outputs[15]
    xp.copyto(outputs[17], i1)
    o17 = outputs[17]
    xp.copyto(outputs[32], i14)
    o32 = outputs[32]
    xp.copyto(outputs[40], i3)
    o40 = outputs[40]
    xp.copyto(outputs[44], i12)
    o44 = outputs[44]
    xp.copyto(outputs[46], i6)
    o46 = outputs[46]
    return outputs

def _k48_r48_p_out(inputs, outputs, xp):
    """4x4x4_48_204_P.slp: 90 adds/subtracts + 0 scalings + 0 negations."""
    i0 = inputs[0]
    i1 = inputs[1]
    i2 = inputs[2]
    i3 = inputs[3]
    i4 = inputs[4]
    i5 = inputs[5]
    i6 = inputs[6]
    i7 = inputs[7]
    i8 = inputs[8]
    i9 = inputs[9]
    i10 = inputs[10]
    i11 = inputs[11]
    i12 = inputs[12]
    i13 = inputs[13]
    i14 = inputs[14]
    i15 = inputs[15]
    i16 = inputs[16]
    i17 = inputs[17]
    i18 = inputs[18]
    i19 = inputs[19]
    i20 = inputs[20]
    i21 = inputs[21]
    i22 = inputs[22]
    i23 = inputs[23]
    i24 = inputs[24]
    i25 = inputs[25]
    i26 = inputs[26]
    i27 = inputs[27]
    i28 = inputs[28]
    i29 = inputs[29]
    i30 = inputs[30]
    i31 = inputs[31]
    i32 = inputs[32]
    i33 = inputs[33]
    i34 = inputs[34]
    i35 = inputs[35]
    i36 = inputs[36]
    i37 = inputs[37]
    i38 = inputs[38]
    i39 = inputs[39]
    i40 = inputs[40]
    i41 = inputs[41]
    i42 = inputs[42]
    i43 = inputs[43]
    i44 = inputs[44]
    i45 = inputs[45]
    i46 = inputs[46]
    i47 = inputs[47]
    z25 = xp.subtract(i21, i35)
    z40 = xp.add(i43, i9)
    z22 = xp.add(i12, i9)
    z33 = xp.add(i31, i19)
    z21 = xp.subtract(i11, i40)
    z43 = xp.subtract(i47, z25)
    z52 = xp.subtract(z25, z40)
    z36 = xp.subtract(i37, z33)
    z49 = xp.add(z33, z21)
    z46 = xp.subtract(z52, z49)
    z27 = xp.subtract(i23, z46)
    z44 = xp.add(i8, z27)
    z50 = xp.subtract(z49, z44)
    z30 = xp.subtract(i27, i25)
    z34 = xp.subtract(i34, z50)
    z41 = xp.add(i45, z34)
    z24 = xp.add(i16, z50)
    z32 = xp.add(i3, z30)
    z45 = xp.add(z52, z30)
    z29 = xp.add(i26, z24)
    z42 = xp.add(i46, z24)
    z31 = xp.add(i29, z32)
    z23 = xp.subtract(z32, i13)
    z10 = xp.subtract(xp.subtract(i18, z40), i17)
    z17 = xp.subtract(xp.subtract(z44, z22), i33)
    o15 = xp.add(xp.add(i15, i40), z22, out=outputs[15])
    z28 = xp.subtract(i24, z45)
    z37 = xp.add(i39, z45)
    z48 = xp.subtract(z42, z10)
    z51 = xp.subtract(xp.subtract(z10, z23), z46)
    z35 = xp.subtract(i36, z36)
    z26 = xp.add(i22, z36)
    z38 = xp.subtract(i4, z28)
    z19 = xp.add(xp.add(i28, z27), z48)
    z11 = xp.add(xp.subtract(i0, i25), z51)
    z18 = xp.add(xp.add(xp.subtract(i14, z21), z48), z51)
    z39 = xp.add(i42, z18)
    z20 = xp.add(xp.add(i2, z18), z22)
    z12 = xp.subtract(xp.subtract(z51, z17), i20)
    z13 = xp.add(xp.subtract(i1, z48), z37)
    z47 = xp.subtract(z19, z20)
    z14 = xp.subtract(z39, z11)
    o8 = xp.add(z39, z41, out=outputs[8])
    o6 = xp.add(z19, z29, out=outputs[6])
    z15 = xp.subtract(z35, z29)
    o1 = xp.subtract(z26, z12, out=outputs[1])
    o0 = xp.subtract(z11, z31, out=outputs[0])
    z16 = xp.subtract(z31, z26)
    o13 = xp.add(xp.add(xp.add(i5, i17), z12), z13, out=outputs[13])
    o5 = xp.subtract(z13, z35, out=outputs[5])
    o11 = xp.subtract(xp.add(xp.subtract(xp.subtract(i7, i35), z22), z41), z38, out=outputs[11])
    o9 = xp.subtract(xp.add(xp.subtract(xp.subtract(xp.subtract(i10, i19), z41), z12), z13), z38, out=outputs[9])
    o10 = xp.add(z20, z38, out=outputs[10])
    o14 = xp.add(xp.add(xp.add(i32, z42), z28), z47, out=outputs[14])
    o12 = xp.add(xp.add(xp.add(z23, z34), i44), z14, out=outputs[12])
    o7 = xp.add(xp.subtract(xp.add(i30, z46), z37), z15, out=outputs[7])
    o4 = xp.add(xp.add(xp.add(i38, z14), z15), z43, out=outputs[4])
    o2 = xp.add(xp.subtract(xp.subtract(i6, z47), z43), z16, out=outputs[2])
    o3 = xp.subtract(xp.subtract(z17, z16), i41, out=outputs[3])
    return outputs

def _kv_r48_matmul(left, right, levels, threshold, schedule):
    """One owned rank48 encoding, one batched child, owned final destinations."""
    if levels < 2 or min(left.shape[-2:]+right.shape[-1:]) <= threshold:
        return _winograd_matmul(left, right, levels, threshold, schedule)
    if any(v % 4 for v in left.shape[-2:]+right.shape[-1:]):
        return _winograd_matmul(left, right, levels, threshold, schedule)
    aa = tuple(part for row in fnp.split(left, 4, axis=-2) for part in fnp.split(row, 4, axis=-1))
    bb = tuple(part for row in fnp.split(right, 4, axis=-2) for part in fnp.split(row, 4, axis=-1))
    encoded_a = fnp.empty(left.shape[:-2]+(48, left.shape[-2]//4, left.shape[-1]//4), dtype=left.dtype)
    encoded_b = fnp.empty(right.shape[:-2]+(48, right.shape[-2]//4, right.shape[-1]//4), dtype=right.dtype)
    _k48_r48_l_out(aa, tuple(fnp.unstack(encoded_a, axis=-3)), fnp)
    _k48_r48_r_out(bb, tuple(fnp.unstack(encoded_b, axis=-3)), fnp)
    products = _winograd_matmul(encoded_a, encoded_b, levels-2, threshold, schedule)
    result = fnp.empty(products.shape[:-3]+(left.shape[-2], right.shape[-1]), dtype=products.dtype)
    outputs = tuple(part for row in fnp.split(result, 4, axis=-2) for part in fnp.split(row, 4, axis=-1))
    _k48_r48_p_out(tuple(fnp.unstack(products, axis=-3)), outputs, fnp)
    return result

def _kv_r48_eligible(m, k, n, batch, depth):
    mode = int(KERNEL_CONFIG.get('kv_r48', 0))
    minimum = int(KERNEL_CONFIG.get('kv_r48_min', 128))
    if mode == 0 or batch != 1 or depth < 2 or min(m, k, n) < minimum or m*k*n < int(KERNEL_CONFIG.get('kv_r48_min_work', 0)):
        return False
    if mode == 1:
        return k >= 4 * max(m, n)
    if mode == 2:
        return n >= 4 * max(m, k)
    if mode == 3:
        return max(k, n) >= 4 * m
    return True


def _kv_decode_packed(products, layout, bank=None):
    p1,p2,p3,p4,p5,p6,p7=products
    axis=-(len(layout)+3)
    if bank is not None and int(KERNEL_CONFIG.get('kv_recycle_packed_output',0)):
        common=fnp.add(p1,p6,out=p6)
        lower=fnp.add(common,p7,out=p7)
        fnp.add(p1,p2,out=p2)
        upper=fnp.add(common,p5,out=p6)
        fnp.add(lower,p5,out=p1)
        fnp.add(upper,p3,out=p3)
        fnp.subtract(lower,p4,out=p4)
        position=len(bank.shape)+axis
        selector=(slice(None),)*position+(slice(0,4),)+(slice(None),)*(len(bank.shape)-position-1)
        return bank[selector],(-2,)+layout
    stack_limit=int(KERNEL_CONFIG.get('kv_packed_stack_elements',0))
    if stack_limit>0 and math.prod(p1.shape)<=stack_limit:
        a=fnp.add(p1,p2,out=p2)
        common=fnp.add(p1,p6,out=p6)
        lower=fnp.add(common,p7,out=p7)
        d=fnp.add(lower,p5,out=p1)
        upper=fnp.add(common,p5,out=p5)
        b=fnp.add(upper,p3,out=p3)
        c=fnp.subtract(lower,p4,out=p4)
        return fnp.stack((c,d,a,b),axis=axis),(-2,)+layout
    prefix=p1.shape[:-(len(layout)+2)]
    tail=p1.shape[-(len(layout)+2):]
    result=fnp.empty(prefix+(4,)+tail,dtype=p1.dtype)
    c,d,a,b=fnp.unstack(result,axis=axis)
    fnp.add(p1,p2,out=a)
    common=fnp.add(p1,p6,out=p6)
    lower=fnp.add(common,p7,out=p7)
    fnp.add(lower,p5,out=d)
    upper=fnp.add(common,p5,out=p5)
    fnp.add(upper,p3,out=b)
    fnp.subtract(lower,p4,out=c)
    return result,(-2,)+layout


def _kv_packed_leaf(left,right,levels):
    aa=_kv_leaf_encode(left,True,levels)
    bb=_kv_leaf_encode(right,False,levels)
    value=aa@bb
    layout=()
    for stage in range(levels):
        parts=fnp.unstack(value,axis=-(len(layout)+3))
        order=(3,0,1,2,4,5,6)
        products=tuple(parts[order.index(i)] for i in range(7))
        value,layout=_kv_decode_packed(products,layout,value)
    return value,layout

def _kv_packed_matmul(left,right,levels,threshold,schedule):
    rows,inner=left.shape[-2:];columns=right.shape[-1]
    if levels<=0 or min(rows,inner,columns)<=threshold or rows%2 or inner%2 or columns%2:
        return left@right,()
    caps,leaf,input_threshold,output_threshold,separable_threshold,decoder_price=schedule
    three_limit=int(KERNEL_CONFIG.get('kw_three_leaf_elements',0))
    if levels==3 and three_limit>0 and not (rows%8 or inner%8 or columns%8):
        elements=max(math.prod(left.shape),math.prod(right.shape),_matmul_batch_size(left,right)*rows*columns)//64
        if elements<=three_limit:
            return _kv_packed_leaf(left,right,levels)
    one_limit=int(KERNEL_CONFIG.get('kw_one_level_elements',0))
    if levels==1 and one_limit>0:
        elements=max(math.prod(left.shape),math.prod(right.shape),_matmul_batch_size(left,right)*rows*columns)//4
        if elements<=one_limit:
            return _kv_packed_leaf(left,right,levels)
    if levels==2 and separable_threshold>0 and not (rows%4 or inner%4 or columns%4):
        elements=max(math.prod(left.shape),math.prod(right.shape),_matmul_batch_size(left,right)*rows*columns)//16
        if elements<=separable_threshold:
            return _kv_packed_leaf(left,right,levels)
    a11,a12,a21,a22=_quarters(left)
    b11,b12,b21,b22=_quarters(right)
    m,k,n=rows//2,inner//2,columns//2
    elements=_matmul_batch_size(left,right)*max(m*k,k*n,m*n)
    chunk=max(1,min(7,caps[min(levels,6)-1]//max(elements,1)))
    if levels<=int(leaf):
        chunk=1
        limit=int(KERNEL_CONFIG.get('kw_leaf_pack_elements',0))
        if levels==1 and limit>0 and elements<=limit:
            chunk=max(1,min(7,caps[min(levels,6)-1]//max(elements,1)))
    lp={0:a11,1:a12,3:a22}
    ls={4:('+',a21,a22),5:('-',4,a11),6:('-',a11,a21),2:('-',a12,5)}
    rp={0:b11,1:b21,2:b22}
    rs={4:('-',b12,b11),5:('-',b22,4),6:('-',b22,b12),3:('-',5,b21)}
    lc,rc={},{}
    products=[None]*7
    order=(3,0,1,2,4,5,6)
    layout=None
    bank=None
    for start in range(0,7,chunk):
        keys=order[start:start+chunk]
        if len(keys)==1:
            key=keys[0]
            aa=lp[key] if key in lp else _winograd_sum(key,ls,lc)
            bb=rp[key] if key in rp else _winograd_sum(key,rs,rc)
            products[key],child_layout=_kv_packed_matmul(aa,bb,levels-1,threshold,schedule)
        else:
            counts=_WINOGRAD_GROUP_COUNTS[chunk,start]
            aa=_winograd_slab(lp,ls,keys,lc,left,input_threshold,counts[0])
            bb=_winograd_slab(rp,rs,keys,rc,right,input_threshold,counts[1])
            packed,child_layout=_kv_packed_matmul(aa,bb,levels-1,threshold,schedule)
            if len(keys)==7:bank=packed
            for key,part in zip(keys,fnp.unstack(packed,axis=-(len(child_layout)+3))):
                products[key]=part
        if layout is None:layout=child_layout
        elif layout!=child_layout:raise ValueError('Inconsistent packed child layout')
    return _kv_decode_packed(products,layout,bank)

def _kv_unpack_packed(value,layout):
    if not layout:return value
    prefix=value.shape[:-(len(layout)+2)]
    m,n=value.shape[-2:]
    bits=tuple(d for radix in layout for d in (abs(radix),abs(radix)))
    value=fnp.reshape(value,prefix+bits+(m,n),copy=False)
    b=len(prefix);depth=len(layout)
    if any(radix<0 for radix in layout):
        selector=[slice(None)]*len(value.shape)
        for i,radix in enumerate(layout):
            if radix<0:selector[b+2*i]=slice(None,None,-1)
        value=value[tuple(selector)]
    axes=tuple(range(b))+tuple(b+2*i for i in range(depth))+(b+2*depth,)+tuple(b+2*i+1 for i in range(depth))+(b+2*depth+1,)
    value=fnp.transpose(value,axes)
    divisor=math.prod(abs(radix) for radix in layout)
    return fnp.reshape(value,prefix+(m*divisor,n*divisor),copy=True)

def _kv_r48_packed(left,right,levels,threshold,schedule):
    if levels<2 or min(left.shape[-2:]+right.shape[-1:])<=threshold or any(v%4 for v in left.shape[-2:]+right.shape[-1:]):
        return _kv_packed_matmul(left,right,levels,threshold,schedule)
    aa=tuple(part for row in fnp.split(left,4,axis=-2) for part in fnp.split(row,4,axis=-1))
    bb=tuple(part for row in fnp.split(right,4,axis=-2) for part in fnp.split(row,4,axis=-1))
    encoded_a=fnp.empty(left.shape[:-2]+(48,left.shape[-2]//4,left.shape[-1]//4),dtype=left.dtype)
    encoded_b=fnp.empty(right.shape[:-2]+(48,right.shape[-2]//4,right.shape[-1]//4),dtype=right.dtype)
    _k48_r48_l_out(aa,tuple(fnp.unstack(encoded_a,axis=-3)),fnp)
    _k48_r48_r_out(bb,tuple(fnp.unstack(encoded_b,axis=-3)),fnp)
    products,layout=_kv_packed_matmul(encoded_a,encoded_b,levels-2,threshold,schedule)
    axis=-(len(layout)+3)
    prefix=products.shape[:-(len(layout)+3)]
    tail=products.shape[-(len(layout)+2):]
    result=fnp.empty(prefix+(16,)+tail,dtype=products.dtype)
    _k48_r48_p_out(tuple(fnp.unstack(products,axis=axis)),tuple(fnp.unstack(result,axis=axis)),fnp)
    return result,(4,)+layout

def _kv_packed_root(left,right,levels,threshold,schedule):
    minimum=int(KERNEL_CONFIG.get('kv_packed_depth_min',0))
    if levels<minimum:
        return _winograd_matmul(left,right,levels,threshold,schedule)
    if _kz_eligible(left.shape[-2:]+right.shape[-1:],levels,threshold):
        work=_matmul_batch_size(left,right)*math.prod(left.shape[-2:])*right.shape[-1]
        return _kz_matmul(left,right,_kz_schemes(levels,work,1))
    value,layout=_kv_packed_matmul(left,right,levels,threshold,schedule)
    return _kv_unpack_packed(value,layout)

def _kv_r48_packed_root(left,right,levels,threshold,schedule):
    minimum=int(KERNEL_CONFIG.get('kv_packed_depth_min',0))
    if levels<minimum:
        return _kv_r48_matmul(left,right,levels,threshold,schedule)
    if _kz_eligible(left.shape[-2:]+right.shape[-1:],levels,threshold):
        work=_matmul_batch_size(left,right)*math.prod(left.shape[-2:])*right.shape[-1]
        return _kz_r48_matmul(left,right,_kz_schemes(levels,work,2))
    value,layout=_kv_r48_packed(left,right,levels,threshold,schedule)
    return _kv_unpack_packed(value,layout)


def _kv_leaf_encode(matrix,is_left,levels):
    value=matrix
    order=(3,0,1,2,4,5,6)
    for stage in range(levels):
        a,b,c,d=_quarters(value)
        if is_left:
            plain={0:a,1:b,3:d}
            sums={4:('+',c,d),5:('-',4,a),6:('-',a,c),2:('-',b,5)}
        else:
            plain={0:a,1:c,2:d}
            sums={4:('-',b,a),5:('-',d,4),6:('-',d,b),3:('-',5,c)}
        value=_winograd_slab(plain,sums,order,{},value,0,(3,0,1))
    return value


# In-place packed Winograd (kz) with a per-level scheme and the deepest level axis outermost.  Each operand is
# converted once to a per-level blocked order; the seven operands of every level live in one buffer
# prefix+(7,)*levels+(r, c) whose level axes run from the deepest level (outermost) to the top level (innermost), so
# the deep, data-heavy levels iterate long contiguous runs.  The leaf is one batched matmul and the decode runs in
# place.  A level uses one of two schemes:
#   'w' Winograd:              positions 0:P5 1:P1 2:P3 3:P2 4:P4 5:P6 6:P7; left data 1:5 (index 2c+r), right data
#                              0:4 (index 2r+1-c), output C11/C21/C12/C22 at 1:5; 4+4 encode sums, 7-add decode.
#   'k' alternative basis:     positions 0:P5 1:P3 2:P1 3:P2 4:P7 5:P4 6:P6; left basis A11/A12/A11-A21/A22 at 2:6
#                              (index 2r+c), right basis B12-B11/B22/B11/B21 at 0:4 (index 2(1-c)+r), output
#                              C11/C21/C22-C12/C22-C21 at 2:6; 3+3 encode sums and a 6-add decode, plus one in-place
#                              basis change per operand on the blocked copy and two on the output.
# Sums, products and decode order do not depend on the axis order; all-'w' levels are bitwise _kv_packed_matmul.
_KZ_LEFT = {'w': slice(1, 5), 'k': slice(2, 6)}
_KZ_RIGHT = {'w': slice(0, 4), 'k': slice(0, 4)}
_KZ_OUT = {'w': slice(1, 5), 'k': slice(2, 6)}

def _kz_sync_decode():
    """kz_fused_decode moves the w output region to slots 3..6; the region is the switch the encode and decode read."""
    _KZ_OUT['w'] = slice(3, 7) if int(KERNEL_CONFIG.get('kz_fused_decode', 0)) else slice(1, 5)

def _kz_regions(schemes, regions, below):
    """Data regions of the level axes deeper than `below`, deepest first (the outer axes)."""
    return tuple(regions[schemes[level]] for level in range(len(schemes) - 1, below, -1))

def _kz_view(buffer, base, schemes, level, regions):
    """The seven planes of `level`; its deeper (outer) level axes restricted to their data regions."""
    deeper = _kz_regions(schemes, regions, level)
    view = buffer[(slice(None),) * base + deeper] if deeper else buffer
    return fnp.unstack(view, axis=base + len(schemes) - 1 - level)

def _kz_blocked(matrix, schemes, right):
    """prefix+(R, C) -> prefix+(4,)*levels+(r, c), deepest level first, in each level's blocked index order."""
    levels = len(schemes)
    prefix = matrix.shape[:-2]
    p = len(prefix)
    r, c = matrix.shape[-2] >> levels, matrix.shape[-1] >> levels
    split = fnp.reshape(matrix, prefix + (2,) * levels + (r,) + (2,) * levels + (c,))
    if right:
        split = split[(slice(None),) * (p + levels + 1) + (slice(None, None, -1),) * levels]
    pairs = ()
    for level in range(levels - 1, -1, -1):
        row, column = p + level, p + levels + 1 + level
        pairs += (row, column) if (schemes[level] == 'w') == right else (column, row)
    axes = tuple(range(p)) + pairs + (p + levels, p + 2 * levels + 1)
    return fnp.reshape(fnp.transpose(split, axes), prefix + (4,) * levels + (r, c))

def _sl_kz_unblocked_original(blocked, levels):
    """prefix+(4,)*levels+(r, c), deepest level first with level index 2c+r -> prefix+(R, C)."""
    prefix = blocked.shape[:-(levels + 2)]
    p = len(prefix)
    (r, c) = blocked.shape[-2:]
    split = fnp.reshape(blocked, prefix + (2,) * (2 * levels) + (r, c))
    axes = tuple(range(p)) + tuple((p + 2 * (levels - 1 - l) + 1 for l in range(levels))) + (p + 2 * levels,) + tuple((p + 2 * (levels - 1 - l) for l in range(levels))) + (p + 2 * levels + 1,)
    return fnp.reshape(fnp.transpose(split, axes), prefix + (r << levels, c << levels))

_SL_KZ_UNPACK_REMAINING = 0

def _sl_kz_reset_unpack():
    """Reset the per-predict fresh-block unpack budget (called by Estimator._predict_impl after the runtime is active)."""
    global _SL_KZ_UNPACK_REMAINING
    _SL_KZ_UNPACK_REMAINING = int(KERNEL_CONFIG['sl_unpack_call_budget'])

def _sl_kz_unstack_tree(value, axis, levels):
    if levels == 0:
        return value
    return tuple(_sl_kz_unstack_tree(part, axis, levels - 1)
                 for part in fnp.unstack(value, axis=axis))

def _sl_kz_tree_leaf(tree, row, column, levels):
    for shift in range(levels):
        tree = tree[((row >> shift) & 1) + 2 * ((column >> shift) & 1)]
    return tree

def _kz_unblocked(blocked, levels):
    global _SL_KZ_UNPACK_REMAINING
    extra_calls = (4 ** levels - 1) // 3 - 2
    if (levels == int(KERNEL_CONFIG['sl_unpack_levels']) and _SL_KZ_UNPACK_REMAINING >= extra_calls
            and math.prod(blocked.shape) >= int(KERNEL_CONFIG['sl_unpack_min_entries'])):
        _SL_KZ_UNPACK_REMAINING -= extra_calls
        axis = len(blocked.shape) - levels - 2
        tree = _sl_kz_unstack_tree(blocked, axis, levels)
        return fnp.block([[_sl_kz_tree_leaf(tree, row, column, levels)
                           for column in range(1 << levels)]
                          for row in range(1 << levels)])
    return _sl_kz_unblocked_original(blocked, levels)

def _kz_basis(blocked, base, schemes, first, right):
    """Alternative-basis change on the blocked copy for 'k' levels >= first: A21 <- A11-A21, B12 <- B12-B11."""
    for level in range(first, len(schemes)):
        if schemes[level] == 'k':
            head = (slice(None),) * (base + len(schemes) - 1 - level)
            first_slot, third_slot = blocked[head + (0,)], blocked[head + (2,)]
            fnp.subtract(first_slot, third_slot, out=first_slot if right else third_slot)

def _kz_output(blocked, base, schemes, first):
    """Map 'k' levels' decoded bases (C11, C21, C22-C12, C22-C21) back to (C11, C21, C12, C22)."""
    for level in range(first, len(schemes)):
        if schemes[level] == 'k':
            head = (slice(None),) * (base + len(schemes) - 1 - level)
            c22, c12 = blocked[head + (3,)], blocked[head + (2,)]
            fnp.add(c22, blocked[head + (1,)], out=c22)
            fnp.subtract(c22, c12, out=c12)

def _kz_left_level(q, scheme):
    if scheme == 'w':
        fnp.add(q[2], q[4], out=q[0])
        fnp.subtract(q[1], q[2], out=q[6])
        fnp.subtract(q[0], q[1], out=q[5])
        fnp.subtract(q[3], q[5], out=q[2])
    else:
        fnp.subtract(q[5], q[4], out=q[6])
        fnp.add(q[6], q[2], out=q[0])
        fnp.subtract(q[3], q[6], out=q[1])

def _kz_right_level(q, scheme):
    if scheme == 'w':
        fnp.subtract(q[2], q[0], out=q[6])
        fnp.subtract(q[0], q[1], out=q[0])
        fnp.subtract(q[2], q[0], out=q[5])
        if _KZ_OUT['w'].start == 3:
            fnp.subtract(q[3], q[5], out=q[4])
        else:
            fnp.subtract(q[5], q[3], out=q[4])
    else:
        fnp.subtract(q[1], q[0], out=q[6])
        fnp.subtract(q[6], q[3], out=q[5])
        fnp.subtract(q[6], q[2], out=q[4])

def _kz_encode_levels(buffer, base, schemes, first, right):
    regions = _KZ_RIGHT if right else _KZ_LEFT
    for level in range(first, len(schemes)):
        q = _kz_view(buffer, base, schemes, level, regions)
        (_kz_right_level if right else _kz_left_level)(q, schemes[level])

def _kz_encode_left(blocked, buffer, base, schemes):
    """Level 0 from the blocked copy into the buffer (plain bases copied), then the deeper levels in place."""
    _kz_basis(blocked, base, schemes, 1, False)
    levels = len(schemes)
    top, head = base + levels - 1, (slice(None),) * base
    deeper = _kz_regions(schemes, _KZ_LEFT, 0)
    full = (slice(None),) * (levels - 1)
    a = fnp.unstack(blocked, axis=top)
    out = fnp.unstack(buffer[head + deeper] if deeper else buffer, axis=top)
    if schemes[0] == 'w':
        fnp.copyto(out[1], a[0])
        fnp.copyto(buffer[head + deeper + (slice(3, 5),)], blocked[head + full + (slice(2, 4),)])
        fnp.add(a[1], a[3], out=out[0])
        fnp.subtract(a[0], a[1], out=out[6])
        fnp.subtract(out[0], a[0], out=out[5])
        fnp.subtract(a[2], out[5], out=out[2])
    else:
        fnp.copyto(buffer[head + deeper + (slice(2, 4),)], blocked[head + full + (slice(0, 2),)])
        fnp.copyto(out[5], a[3])
        fnp.subtract(a[0], a[2], out=out[4])
        fnp.subtract(a[3], out[4], out=out[6])
        fnp.add(out[6], a[0], out=out[0])
        fnp.subtract(a[1], out[6], out=out[1])
    _kz_encode_levels(buffer, base, schemes, 1, False)

def _kz_encode_right(blocked, buffer, base, schemes):
    _kz_basis(blocked, base, schemes, 1, True)
    levels = len(schemes)
    (top, head) = (base + levels - 1, (slice(None),) * base)
    deeper = _kz_regions(schemes, _KZ_RIGHT, 0)
    full = (slice(None),) * (levels - 1)
    b = fnp.unstack(blocked, axis=top)
    out = fnp.unstack(buffer[head + deeper] if deeper else buffer, axis=top)
    fnp.copyto(buffer[head + deeper + (slice(1, 4),)], blocked[head + full + (slice(1, 4),)])
    if schemes[0] == 'w':
        fnp.subtract(b[2], b[0], out=out[6])
        fnp.subtract(b[0], b[1], out=out[0])
        fnp.subtract(b[2], out[0], out=out[5])
        if _KZ_OUT['w'].start == 3:
            fnp.subtract(b[3], out[5], out=out[4])
        else:
            fnp.subtract(out[5], b[3], out=out[4])
    else:
        fnp.subtract(b[0], b[2], out=out[0])
        fnp.subtract(b[1], out[0], out=out[6])
        fnp.subtract(out[6], b[3], out=out[5])
        fnp.subtract(out[6], b[2], out=out[4])
    _kz_encode_levels(buffer, base, schemes, 1, True)

def _kz_decode(products, base, schemes):
    for level in range(len(schemes) - 1, -1, -1):
        if schemes[level] == 'w' and _KZ_OUT['w'].start == 3:
            deeper = _kz_regions(schemes, _KZ_OUT, level)
            view = products[(slice(None),) * base + deeper] if deeper else products
            axis = base + len(schemes) - 1 - level
            head = (slice(None),) * axis
            q = fnp.split(view, 7, axis=axis)
            u2_c11 = view[head + (slice(5, 2, -2),)]
            fnp.add(q[1], u2_c11, out=u2_c11)
            u3_u4 = view[head + (slice(6, 0, -5),)]
            fnp.add(q[5], view[head + (slice(6, None, -6),)], out=u3_u4)
            fnp.add(u3_u4, view[head + (slice(4, 1, -2),)], out=view[head + (slice(4, 6),)])
            fnp.add(q[6], q[0], out=q[6])
            continue
        q = _kz_view(products, base, schemes, level, _KZ_OUT)
        if schemes[level] == 'w':
            fnp.add(q[1], q[5], out=q[5])
            fnp.add(q[1], q[3], out=q[1])
            fnp.add(q[5], q[6], out=q[6])
            fnp.add(q[5], q[0], out=q[5])
            fnp.add(q[5], q[2], out=q[3])
            fnp.subtract(q[6], q[4], out=q[2])
            fnp.add(q[6], q[0], out=q[4])
        else:
            fnp.add(q[2], q[6], out=q[6])
            fnp.add(q[2], q[3], out=q[2])
            fnp.add(q[6], q[4], out=q[6])
            fnp.subtract(q[6], q[5], out=q[3])
            fnp.subtract(q[4], q[1], out=q[4])
            fnp.add(q[0], q[5], out=q[5])

def _kz_matmul(left, right, schemes):
    """Packed product with one level per scheme letter; matrix dims divisible by 2**len(schemes)."""
    levels = len(schemes)
    m, k, n = left.shape[-2] >> levels, left.shape[-1] >> levels, right.shape[-1] >> levels
    left_prefix, right_prefix = left.shape[:-2], right.shape[:-2]
    encoded_left = fnp.empty(left_prefix + (7,) * levels + (m, k), dtype=left.dtype)
    _kz_encode_left(_kz_blocked(left, schemes, False), encoded_left, len(left_prefix), schemes)
    encoded_right = fnp.empty(right_prefix + (7,) * levels + (k, n), dtype=right.dtype)
    _kz_encode_right(_kz_blocked(right, schemes, True), encoded_right, len(right_prefix), schemes)
    products = fnp.matmul(encoded_left, encoded_right)
    base = len(products.shape) - levels - 2
    _kz_decode(products, base, schemes)
    result = products[(slice(None),) * base + _kz_regions(schemes, _KZ_OUT, -1)]
    _kz_output(result, base, schemes, 0)
    return _kz_unblocked(result, levels)

def _kz_blocks16(blocked, axis, right, scheme_pair):
    """The 16 row-major 4x4 blocks (index 4*bi+bj); level-1 axis at `axis`, the level-0 axis right after it."""
    planes = [fnp.unstack(part, axis=axis) for part in fnp.unstack(blocked, axis=axis)]
    def index(level, row, column):
        if right:
            column = 1 - column
        return 2 * row + column if (scheme_pair[level] == 'w') == right else 2 * column + row
    return tuple(planes[index(1, bi & 1, bj & 1)][index(0, bi >> 1, bj >> 1)] for bi in range(4) for bj in range(4))

def _kz_r48_matmul(left, right, schemes):
    """Rank-48 top (the first two scheme letters only order its blocks) over in-place packed levels; the 48 axis sits
    inside the level axes."""
    levels = len(schemes)
    inner = schemes[2:]
    depth = len(inner)
    m, k, n = left.shape[-2] >> levels, left.shape[-1] >> levels, right.shape[-1] >> levels
    left_prefix, right_prefix = left.shape[:-2], right.shape[:-2]
    lp, rp = len(left_prefix), len(right_prefix)
    blocked_left, blocked_right = _kz_blocked(left, schemes, False), _kz_blocked(right, schemes, True)
    _kz_basis(blocked_left, lp, schemes, 2, False)
    _kz_basis(blocked_right, rp, schemes, 2, True)
    encoded_left = fnp.empty(left_prefix + (7,) * depth + (48,) + (m, k), dtype=left.dtype)
    encoded_right = fnp.empty(right_prefix + (7,) * depth + (48,) + (k, n), dtype=right.dtype)
    left_regions, right_regions = _kz_regions(inner, _KZ_LEFT, -1), _kz_regions(inner, _KZ_RIGHT, -1)
    _k48_r48_l_out(_kz_blocks16(blocked_left, lp + depth, False, schemes[:2]),
                   tuple(fnp.unstack(encoded_left[(slice(None),) * lp + left_regions] if depth else encoded_left, axis=lp + depth)), fnp)
    _k48_r48_r_out(_kz_blocks16(blocked_right, rp + depth, True, schemes[:2]),
                   tuple(fnp.unstack(encoded_right[(slice(None),) * rp + right_regions] if depth else encoded_right, axis=rp + depth)), fnp)
    _kz_encode_levels(encoded_left, lp, inner, 0, False)
    _kz_encode_levels(encoded_right, rp, inner, 0, True)
    products = fnp.matmul(encoded_left, encoded_right)
    base = len(products.shape) - depth - 3
    _kz_decode(products, base, inner)
    result = fnp.empty(products.shape[:base] + (4,) * depth + (4, 4) + (m, n), dtype=products.dtype)
    out_regions = _kz_regions(inner, _KZ_OUT, -1)
    _k48_r48_p_out(tuple(fnp.unstack(products[(slice(None),) * base + out_regions] if depth else products, axis=base + depth)),
                   _kz_blocks16(result, base + depth, False, 'ww'), fnp)
    _kz_output(result, base, schemes, 2)
    return _kz_unblocked(result, levels)

def _kz_eligible(dims, levels, threshold):
    """kz_inplace_levels gate; the packed recursion must reach all `levels` (even dims above the leaf threshold)."""
    minimum = int(KERNEL_CONFIG.get('kz_inplace_levels', 0))
    return (minimum > 0 and levels >= minimum and all(v % (1 << levels) == 0 for v in dims) and
            min(dims) >> (levels - 1) > threshold)

def _kz_schemes(levels, work, top):
    """Winograd levels, with the deepest kz_alt_levels in the alternative basis for roots of at least
    kz_alt_min_work batch*m*k*n; the top `top` levels (rank-48 or the root) stay Winograd."""
    alt = int(KERNEL_CONFIG.get('kz_alt_levels', 0))
    if work < int(KERNEL_CONFIG.get('kz_alt_min_work', 0)):
        alt = 0
    alt = max(0, min(alt, levels - top))
    return 'w' * (levels - alt) + 'k' * alt

def _leg_transport_schedule(m, k, n, batch, depth, leaf, multiple, schedule):
    """Use one more exact Winograd level for aligned, wide leg transports.

    A 16-wide leaf can be subdivided into 8-wide leaves to save products.
    Keep the packed prefix unchanged by evaluating the extra level
    unpacked. Limit the aspect ratio to bound the added operation count.
    All state is local; the supplied numerical kernels are reused.
    """
    if (batch != 1 or depth < 3 or n < 4 * max(m, k)
            or int(KERNEL_CONFIG.get('kv_r48', 0)) != 2
            or not int(KERNEL_CONFIG.get('kv_packed_output', 0))
            or int(KERNEL_CONFIG.get('kw_classical_plan', 0))
            or schedule[1] != 1):
        return depth, schedule
    pm, pk, pn = m, k, n
    if multiple > 0:
        padded = tuple((v + multiple - 1) // multiple * multiple for v in (m, k, n))
        if math.prod(padded) * 10000 <= m * k * n * int(KERNEL_CONFIG['ck_pad_limit']):
            pm, pk, pn = padded
    divisor = 1 << (depth + 1)
    if (min(pm, pk) < leaf * divisor or not 4 * max(pm, pk) <= pn <= 9 * max(pm, pk)
            or any(v % divisor for v in (pm, pk, pn))
            or not _kv_r48_eligible(m, k, n, batch, depth)):
        return depth, schedule
    return depth + 1, (schedule[0], schedule[1] + 1) + schedule[2:]


_KP_R40_GRAPHS = {(3, 3, 6): (((('add', 0, (1, 2, 3, 5), (-8, -6, -7, -9), (-5, -3, -1, -3), False, False),
               ('sub', 0, (0, 6, 14, 17, 19), (-7, 1, 3, -2, 1), (-1, 2, 5, -8, -4), False, False),
               ('add', 0, (4, 9, 13, 18), (0, -8, 3, 0), (1, 2, 5, 17), False, False),
               ('sub', 0, (7, 8, 15, 20), (6, 4, -9, 19), (0, 2, 0, -7), False, False),
               ('add', 0, (10, 12, 25), (3, 2, 7), (9, 4, 18), False, False),
               ('sub', 0, (11, 16, 21, 24, 45), (10, 15, 12, 7, 25), (-5, -3, 8, 20, 8), False, False),
               ('sub', 0, (22, 23, 26), (11, 7, 21), (13, 16, 11), False, False),
               ('sub', 0, (28, 29, 31), (23, 22, 23), (14, 14, 8), False, False),
               ('add',
                0,
                (27, 30, 35, 36, 37, 41, 43, 44, 46, 51, 52, 53),
                (12, 16, 14, 13, 24, 11, 24, 21, 21, 12, 7, 25),
                (16, 22, 26, 23, 29, 25, 26, 25, 24, 24, 21, 29),
                False,
                False),
               ('sub', 0, (32, 33, 42, 47, 48), (27, 12, 31, 28, 21), (13, 28, 22, 20, 29), False, False),
               ('sub', 0, (34, 50), (33, 32), (11, 20), False, False),
               ('add', 0, (38, 39, 40), (14, 18, 16), (27, 30, 35), False, False),
               ('sub', 0, (49,), (34,), (18,), True, True)),
              (7,
               42,
               48,
               38,
               8,
               32,
               30,
               26,
               28,
               34,
               11,
               12,
               52,
               36,
               40,
               29,
               31,
               13,
               14,
               16,
               45,
               53,
               41,
               18,
               51,
               20,
               37,
               43,
               23,
               27,
               35,
               22,
               44,
               39,
               25,
               49,
               47,
               46,
               24,
               50),
              ((-6,),
               (-1, -2, -4),
               (-8, 1, 5, 17),
               (-7, -9, 0, 6, 19),
               (3, 2, 4, 9),
               (-5, -3, 10, 15),
               (),
               (),
               (),
               (21,),
               (33,),
               (),
               ())),
             ((('add',
                0,
                (0, 1, 2, 6, 9, 12, 14, 25, 60),
                (-14, -15, -18, -8, -10, -11, -9, -5, -2),
                (-1, -5, -6, -7, -3, -9, -4, -4, -1),
                False,
                False),
               ('sub',
                0,
                (3, 4, 5, 11, 13, 16, 34, 39),
                (-8, -6, -9, -14, -13, -10, -3, -14),
                (2, -18, -11, -17, -15, -2, -11, -8),
                False,
                False),
               ('sub',
                0,
                (15, 17, 20, 30, 31, 35, 36, 40),
                (14, 16, -16, 4, 5, 34, -7, 39),
                (3, -13, 13, 6, 6, -16, 3, -12),
                False,
                False),
               ('add',
                0,
                (7, 10, 21, 22, 24, 26, 33, 37, 59, 61),
                (-12, -17, 1, -8, 11, 2, -16, 4, 25, -7),
                (1, 9, 17, 12, 15, 5, 13, 12, 35, 60),
                False,
                False),
               ('sub',
                0,
                (18, 19, 23, 28, 38, 41, 48, 55, 62),
                (15, 10, 22, 17, 25, 40, 26, 24, 61),
                (11, 0, -7, 1, 35, -13, 21, 20, 4),
                False,
                False),
               ('add',
                0,
                (8, 29, 42, 44, 47, 50, 51, 52, 63),
                (0, 0, 38, 19, 21, 28, 20, 18, 59),
                (7, 10, 41, 36, 26, 37, 24, 33, 62),
                False,
                False),
               ('add', 0, (32, 45, 65, 101, 108), (-10, 29, 42, 44, 50), (8, 30, 48, 48, 63), False, False),
               ('sub',
                0,
                (27, 43, 46, 49, 53, 56, 69, 72, 102, 106),
                (-10, 19, 29, 28, 31, 18, 65, 50, 101, 47),
                (8, 36, 30, 37, 32, 33, 51, 45, 52, 51),
                False,
                False),
               ('sub',
                0,
                (58, 70, 71, 73, 107, 109),
                (23, 69, 46, 48, 106, 108),
                (27, 52, 49, 53, 43, 46),
                False,
                False),
               ('add',
                0,
                (54, 57, 66, 75, 80, 84, 91),
                (31, 23, 44, 43, 44, 47, 46),
                (32, 27, 58, 65, 70, 70, 53),
                False,
                False),
               ('sub',
                0,
                (64, 68, 78, 81, 85, 92, 104),
                (57, 66, 70, 47, 47, 91, 43),
                (43, 56, 45, 54, 66, 75, 71),
                False,
                False),
               ('sub', 0, (67, 83, 95, 97, 98), (64, 71, 50, 68, 48), (55, 68, 64, 73, 64), False, False),
               ('add',
                0,
                (74, 76, 86, 87, 96, 99, 105),
                (44, 63, 71, 66, 65, 72, 73),
                (54, 68, 85, 70, 95, 98, 104),
                False,
                False),
               ('add', 0, (77, 79, 88, 93), (67, 74, 49, 72), (76, 78, 87, 74), False, False),
               ('add', 0, (89,), (45,), (77,), True, True),
               ('sub', 0, (82, 90, 94, 100, 103), (81, 89, 93, 72, 71), (67, 46, 47, 67, 77), False, False)),
              (67,
               42,
               84,
               68,
               43,
               44,
               45,
               46,
               100,
               80,
               75,
               83,
               47,
               48,
               49,
               50,
               51,
               52,
               109,
               103,
               105,
               94,
               53,
               54,
               97,
               82,
               79,
               92,
               55,
               56,
               88,
               96,
               86,
               99,
               57,
               58,
               90,
               107,
               63,
               102),
              ((-5, -1, -4),
               (-14, -15, -18, -11, -9, -2, -6, -3),
               (14, 16, 34, 39, 3, 6),
               (-8, 2, -17, -16, 5, 13, -12, 9, 12, 60),
               (-7, -13, 4, 1, 11, 25, 17, 15, 35, 22, 40, 61),
               (10, 26, 24, 0, 21, 20, 38, 59, 7, 41, 62),
               (),
               (-10, 19, 28, 18, 36, 37, 33, 29, 8, 30, 101),
               (69, 106, 108),
               (31, 32, 23, 27),
               (91,),
               (64,),
               (65, 70, 66, 73, 85, 95, 98, 104),
               (74, 76, 78, 87),
               (),
               (71, 72, 77, 81, 89, 93))),
             ((('add',
                0,
                (2, 3, 6, 21, 26, 28, 30, 32, 36, 48, 56, 59, 63),
                (-18, -31, -21, -39, -30, -24, -22, -36, -14, -23, -21, -39, -19),
                (-6, -27, -17, -35, -25, -20, -18, -10, -9, -4, -2, -1, -16),
                False,
                False),
               ('sub',
                0,
                (0, 1, 4, 5, 7, 10, 14, 19, 24, 33, 45, 52, 64),
                (-13, -19, -11, -12, -26, -14, 3, -37, 3, 32, -5, -3, 63),
                (-17, -23, -32, -30, -29, 2, -15, -34, -28, -31, -38, -38, -39),
                False,
                False),
               ('add',
                0,
                (11, 25, 29, 31, 34, 37, 49, 53, 57, 60),
                (7, -32, 6, 1, -15, -34, -30, -18, -32, -20),
                (10, 24, 28, 30, 33, 36, 48, 52, 56, 59),
                False,
                False),
               ('sub',
                0,
                (8, 13, 27, 35, 38, 46, 50, 75),
                (-21, 11, 7, 34, 37, 45, 49, 31),
                (5, -36, 26, 6, -29, 7, -36, 29),
                False,
                False),
               ('add',
                0,
                (15, 18, 39, 47, 54, 61, 68, 80),
                (13, -40, 1, 0, -27, -26, 29, 25),
                (14, 11, 38, 46, 53, 60, 31, 27),
                False,
                False),
               ('sub',
                0,
                (9, 20, 51, 55, 67, 71),
                (8, 19, 50, 54, 25, 39),
                (0, 13, 2, 4, 27, 35),
                False,
                False),
               ('add',
                0,
                (12, 23, 42, 58, 62, 70, 79),
                (-38, -33, -7, 15, 5, 35, 18),
                (9, 9, 15, 57, 61, 39, 20),
                False,
                False),
               ('sub',
                0,
                (16, 22, 66, 77, 78, 90, 93),
                (12, 21, 20, 62, 58, 75, 79),
                (-23, 12, 18, 55, 51, 70, 80),
                False,
                False),
               ('add',
                0,
                (17, 43, 72, 74, 76, 95),
                (4, -34, 51, 22, 55, 79),
                (16, 42, 58, 23, 62, 80),
                False,
                False),
               ('sub',
                0,
                (44, 73, 84, 96, 101, 112),
                (43, 22, 66, 95, 77, 71),
                (-20, 23, 67, 78, 72, 76),
                False,
                False),
               ('add',
                0,
                (40, 65, 83, 91, 108, 113, 124),
                (-8, 17, 44, 68, 72, 96, 73),
                (17, 64, 66, 74, 77, 112, 90),
                False,
                False),
               ('sub',
                0,
                (41, 69, 92, 94, 100, 102, 106, 128),
                (40, 47, 91, 75, 73, 101, 93, 84),
                (3, 44, 76, 73, 90, 84, 65, 65),
                False,
                False),
               ('add',
                0,
                (81, 87, 88, 97, 107, 109, 116, 120),
                (41, 41, 65, 65, 71, 94, 78, 65),
                (44, 69, 74, 76, 106, 108, 92, 100),
                False,
                False),
               ('sub',
                0,
                (85, 89, 98, 104, 117, 118, 126, 130, 131),
                (41, 88, 97, 87, 116, 92, 69, 70, 94),
                (47, 68, 78, 93, 71, 78, 41, 102, 65),
                False,
                False),
               ('add',
                0,
                (82, 86, 111, 127, 129, 132),
                (47, 67, 71, 109, 109, 102),
                (81, 85, 89, 126, 128, 131),
                False,
                False),
               ('sub',
                0,
                (99, 103, 105, 110, 119),
                (71, 86, 104, 82, 118),
                (82, 83, 65, 89, 87),
                False,
                False),
               ('add', 0, (114, 121, 125), (82, 83, 103), (96, 86, 124), False, False),
               ('add', 0, (115, 122), (76, 100), (114, 121), False, False),
               ('sub', 0, (123,), (65,), (103,), True, True)),
              (132, 111, 119, 117, 125, 110, 129, 115, 107, 105, 122, 113, 99, 120, 127, 123, 98, 130),
              ((-24, -22, -6, -35, -25, -10, -9, -4, -2, -1, -16),
               (-31, -39, -14, -19, -17, -13, -11, -12, -37, 32, -5, -3, 63, -28),
               (-18, -30, -32, -15, 10, 24, 28, 30, 33, 36, 48, 52, 56, 59),
               (-21, -36, -29, 7, 6, 34, 37, 45, 49, 26),
               (-27, -26, 1, 11, 31, 29, -40, 14, 38, 46, 53, 60),
               (2, 13, 0, 25, 27, 8, 19, 50, 54),
               (-38, 5, 39, 35, -33, -7, 15, 9, 57, 61),
               (-23, 18, 20, 12, 21),
               (-34, 4, 62, 58, 79, 55, 51, 80, 16, 42),
               (-20, 22, 23, 43, 95),
               (66, 77, 72, -8, 17, 64, 112),
               (3, 75, 73, 90, 40, 91, 101, 84),
               (44, 74, 106, 108),
               (70, 78, 68, 93, 41, 94, 69, 92, 88, 97, 116),
               (67, 47, 102, 109, 81, 85, 126, 128, 131),
               (71, 87, 89, 104, 118),
               (96, 86, 82, 83, 124),
               (76, 100, 114, 121),
               (65, 103)))),
 (3, 6, 3): (((('sub',
                0,
                (0, 1, 2, 3, 4, 6, 12, 13, 14, 15),
                (-12, -1, -2, -7, -7, -9, -2, -4, -14, -13),
                (-18, -16, -10, -14, -8, -11, -5, -13, -17, -15),
                False,
                False),
               ('add',
                0,
                (5, 7, 8, 10, 16, 17, 21, 23, 27, 35, 43, 60),
                (-5, -15, -9, -3, -18, -3, 4, -11, -6, 1, -14, 14),
                (0, -4, -6, -1, -12, 2, 6, -9, 3, 2, -2, 15),
                False,
                False),
               ('sub',
                0,
                (9, 18, 19, 22, 26, 28, 32, 36, 61),
                (8, 17, 7, -17, 6, 27, 2, 14, 60),
                (-17, -16, 10, 4, 4, -11, 1, 15, 16),
                False,
                False),
               ('add',
                0,
                (11, 24, 29, 30, 34, 37, 44, 54),
                (1, -14, 0, 12, 3, 7, 5, 12),
                (5, 23, 18, 16, 9, 10, 43, 19),
                False,
                False),
               ('sub',
                0,
                (20, 25, 33, 38, 40, 45, 51, 52, 56, 62),
                (9, 24, -10, 19, 22, 44, 32, 29, 12, 61),
                (3, 13, 11, 22, 37, 6, 34, 36, 28, 18),
                False,
                False),
               ('add',
                0,
                (31, 42, 46, 49, 50, 53, 55, 63),
                (-10, 25, 13, 20, 32, 29, 28, 26),
                (11, 30, 45, 35, 34, 36, 54, 33),
                False,
                False),
               ('sub',
                0,
                (39, 41, 47, 48, 57, 59, 64, 65, 70, 74, 107),
                (38, 40, 25, 20, 56, 31, 26, 46, 55, 42, 53),
                (0, 0, 30, 35, 37, 21, 33, 51, 52, 49, 46),
                False,
                False),
               ('sub',
                0,
                (66, 68, 71, 76, 87, 94, 103, 108),
                (57, 65, 70, 63, 46, 47, 39, 107),
                (53, 59, 39, 48, 59, 50, 49, 49),
                False,
                False),
               ('add',
                0,
                (58, 75, 78, 85, 89, 92, 95, 100, 104),
                (21, 47, 48, 46, 41, 42, 55, 47, 64),
                (31, 71, 52, 63, 65, 51, 94, 65, 103),
                False,
                False),
               ('sub',
                0,
                (67, 69, 77, 86, 88, 90, 93, 98),
                (66, 68, 76, 85, 87, 75, 92, 50),
                (41, 58, 41, 71, 39, 48, 57, 68),
                False,
                False),
               ('sub', 0, (72, 73, 82, 91, 99), (62, 69, 74, 69, 98), (67, 50, 67, 39, 41), False, False),
               ('add',
                0,
                (79, 80, 83, 96, 101, 105),
                (73, 42, 64, 64, 76, 72),
                (78, 72, 67, 69, 100, 74),
                False,
                False),
               ('sub', 0, (81, 102, 106), (80, 72, 105), (75, 48, 47), False, False),
               ('add', 0, (84, 97), (73, 74), (83, 96), False, False)),
              (90,
               39,
               41,
               82,
               71,
               67,
               91,
               89,
               42,
               73,
               46,
               47,
               48,
               49,
               50,
               51,
               52,
               53,
               93,
               95,
               55,
               57,
               108,
               79,
               106,
               102,
               58,
               59,
               81,
               62,
               99,
               88,
               77,
               104,
               86,
               84,
               97,
               63,
               101,
               64),
              ((-7, -13, -8),
               (-12, -1, -2, -9, -4, -18, -5, -15, -3, -6),
               (-16, -11, -17, 4, 14, 2, 15, 8, 17, 27, 60),
               (-14, 1, 7, 10, 16, 5, 23, 43),
               (6, 3, 12, 18, 9, 19, 24, 22, 44, 61),
               (-10, 32, 29, 13, 11, 34, 36, 28, 45, 54),
               (0, 37, 25, 20, 26, 30, 35, 33, 38, 40, 56),
               (70, 107),
               (31, 21, 65, 94, 103),
               (66, 68, 85, 87, 92),
               (98,),
               (76, 69, 78, 100),
               (75, 72, 80, 105),
               (74, 83, 96))),
             ((('add',
                0,
                (0, 1, 4, 5, 7, 9, 11, 12, 14, 47, 57),
                (-13, -6, -5, -18, -10, -4, -9, -15, -11, -9, -4),
                (-9, -1, -2, -16, -8, -3, -1, -4, -7, -6, -1),
                False,
                False),
               ('sub',
                0,
                (2, 6, 8, 13, 16, 21, 31),
                (-8, -2, -16, 12, -5, 7, -11),
                (-14, -12, -18, -17, 5, 11, 0),
                False,
                False),
               ('add',
                0,
                (3, 15, 17, 18, 24, 27, 34, 48, 58),
                (-11, -15, -14, 7, 6, 2, 2, -12, -7),
                (0, 14, 8, 11, 13, 4, 5, 47, 57),
                False,
                False),
               ('sub',
                0,
                (19, 22, 23, 25, 28, 30, 32, 40, 49),
                (2, 6, 16, 4, 3, 1, 31, 24, 48),
                (4, 13, -2, 8, 9, 15, 9, 18, -15),
                False,
                False),
               ('add',
                0,
                (10, 26, 29, 36, 39, 42, 43, 50, 59),
                (1, -8, 1, 23, 21, 18, 28, 7, -10),
                (3, 17, 15, 30, 22, 24, 34, 49, 58),
                False,
                False),
               ('sub',
                0,
                (33, 35, 37, 41, 44, 45, 51),
                (-17, 23, 25, 21, 34, 32, 50),
                (10, 30, 29, 22, 28, 26, 16),
                False,
                False),
               ('add',
                0,
                (20, 38, 46, 56, 60, 63, 65, 77, 91, 101, 103),
                (-17, 25, 26, 27, -13, 35, 36, 39, 36, 42, 37),
                (10, 29, 32, 33, 59, 44, 43, 44, 40, 43, 40),
                False,
                False),
               ('sub',
                0,
                (52, 53, 55, 66, 67, 78, 80, 92),
                (51, 19, 27, 38, 41, 77, 38, 41),
                (-3, 20, 33, 45, 63, 56, 41, 91),
                False,
                False),
               ('add',
                0,
                (54, 61, 64, 71, 72, 76, 81, 95, 97),
                (19, 6, 37, 35, 53, 40, 53, 56, 40),
                (20, 60, 46, 52, 66, 65, 80, 67, 52),
                False,
                False),
               ('sub',
                0,
                (62, 68, 70, 79, 87, 89, 93, 94, 96, 98, 102, 108),
                (61, 64, 55, 63, 43, 72, 92, 43, 95, 97, 101, 66),
                (17, 39, 65, 72, 54, 42, 71, 71, 37, 67, 55, 42),
                False,
                False),
               ('add', 0, (69, 83, 99, 104), (46, 54, 63, 54), (62, 64, 64, 103), False, False),
               ('sub',
                0,
                (73, 74, 82, 84, 85, 88, 90, 100, 105),
                (69, 70, 66, 83, 42, 87, 89, 99, 69),
                (38, 38, 70, 65, 62, 68, 44, 56, 39),
                False,
                False),
               ('sub', 0, (75, 86, 106), (74, 85, 105), (40, 68, 42), False, False),
               ('sub', 0, (107,), (106,), (45,), True, True)),
              (68,
               67,
               76,
               108,
               35,
               36,
               37,
               38,
               39,
               40,
               41,
               42,
               43,
               44,
               45,
               46,
               52,
               93,
               73,
               86,
               79,
               84,
               53,
               54,
               90,
               88,
               104,
               81,
               78,
               102,
               75,
               96,
               55,
               56,
               100,
               82,
               107,
               94,
               62,
               98),
              ((-6, -4, -9, -1),
               (-5, -18, -16, 12),
               (-11, -7, -14, -12, 5, 11, 0, 14, 47, 57),
               (-15, -2, 2, 8, 13, 4, 31, 48, 9),
               (-10, -8, 7, 3, 1, 24, 15, 18, 49, 58),
               (16, 23, 21, 28, 30, 22, 34, 50),
               (-13, -17, 25, 32, 10, 29, 26, 59),
               (-3, 27, 33, 51, 77, 91),
               (6, 19, 20, 60, 80),
               (17, 61, 72, 92, 95, 97, 101, 71),
               (63, 64, 103),
               (66, 65, 69, 70, 83, 87, 89, 99),
               (74, 85, 105),
               (106,))),
             ((('sub',
                0,
                (0, 1, 3, 4, 6, 9, 10, 11, 13, 14, 16, 19),
                (-32, -35, -28, -4, -32, -23, -31, -36, -10, -9, -39, -26),
                (-17, -16, -40, -37, -11, -8, -18, -15, -11, -19, -12, -30),
                False,
                False),
               ('add',
                0,
                (2, 5, 12, 20, 26, 30, 35, 40, 44, 48, 55, 60),
                (-38, -20, -24, 4, -34, -29, -39, -22, -21, -14, -5, -27),
                (-27, -1, -7, 19, -7, -20, -37, 11, 1, 10, 0, -3),
                False,
                False),
               ('sub',
                0,
                (7, 15, 18, 27, 31, 36, 41, 45, 53, 61),
                (6, 14, 12, 26, 30, 35, 40, 44, 11, 60),
                (-28, 4, -13, 9, -19, 2, -16, -15, -6, -31),
                False,
                False),
               ('add',
                0,
                (8, 17, 21, 49, 56, 62),
                (-2, 5, 5, 9, 1, -10),
                (7, 16, 20, 48, 55, 61),
                False,
                False),
               ('sub',
                0,
                (22, 23, 32, 37, 42, 50),
                (21, 17, 31, 36, 41, 49),
                (18, 12, -30, 3, 15, 13),
                False,
                False),
               ('add',
                0,
                (28, 33, 46, 51, 54, 57),
                (8, 0, 17, 3, 22, 13),
                (27, 32, 45, 50, 53, 56),
                False,
                False),
               ('add', 0, (29, 34, 47, 52, 58), (15, 10, 8, 22, 2), (28, 33, 46, 51, 57), False, False),
               ('add', 0, (59, 71), (18, 29), (58, 52), False, False),
               ('sub',
                0,
                (24, 38, 64, 66, 67, 72),
                (23, 37, 47, 52, 59, 71),
                (-33, -26, 29, 34, 47, 59),
                False,
                False),
               ('sub', 0, (25, 39), (24, 38), (-8, -25), False, False),
               ('add', 0, (43, 74, 77, 84), (25, 29, 39, 39), (42, 59, 52, 66), False, False),
               ('sub', 0, (63, 65), (62, 64), (25, 43), False, False),
               ('sub', 0, (69,), (63,), (54,), True, True),
               ('add', 0, (68, 70, 73, 75, 78), (54, 43, 69, 66, 67), (63, 69, 72, 74, 77), False, False),
               ('sub', 0, (76, 80), (67, 68), (70, 75), False, False),
               ('add', 0, (79, 81, 85), (70, 39, 76), (78, 68, 84), False, False),
               ('sub', 0, (83,), (52,), (76,), True, True),
               ('add', 0, (82,), (75,), (81,), True, True)),
              (83, 65, 73, 39, 82, 80, 79, 85, 34),
              ((-32, -35, -4, -23, -36, -9, -17, -40, -11, -18, -12),
               (-39, -37, -38, -20, -24, -34, -29, -22, -21, -14, -5, -27, -1, -7, 19, -3),
               (-28, -31, -16, -15, -19, 4, 11, 6, 14, 26, 30, 35, 40, 44, 60, -13, -6),
               (-10, 1, 9, -2, 5, 7, 16, 20, 48, 55, 61),
               (-30, 12, 21, 31, 36, 41, 49),
               (0, 17, 3, 13, 27, 32, 45, 50, 53, 56),
               (10, 2, 15, 8, 22, 28, 33, 46, 51, 57),
               (18, 58),
               (-26, 23, 37, 47, 71, -33),
               (-8, 24, 38, -25),
               (29, 59, 42),
               (25, 62, 64),
               (),
               (66, 43, 63, 54, 69, 72, 74, 77),
               (67,),
               (68, 70, 78, 84),
               (52, 76),
               (75, 81))))}

# Smirnov's exact rank-40 3x3x6 and 3x6x3 formulas, adapted from
# Jean-Guillaume Dumas, Clement Pernet and Alexandre Sedoglavic,
# https://github.com/jgdumas/Fast-Matrix-Multiplication,
# FMMa_3_3_6.m / FMMa_3_6_3.m. CeCILL-B; see LICENSE_CeCILL-B.txt.
# This fresh-array Python adaptation factors signs and powers of two out of
# the input circuits. Its integer tensor identity was checked entry by entry:
# sum_r W[c,r]*U[r,a]*V[r,b] equals 8 times matrix-multiplication incidence.
# The final common scale is 1/8. Only arithmetic order changes.


# Rank-40 <3,3,6> / <3,6,3> top over the in-place packed levels (Dumas-Pernet-Sedoglavic circuits, CeCILL-B; see the
# _KP_R40_GRAPHS notice).  The circuit nodes run one op each; its outputs are written straight into the encoded buffers
# (prefix+(7,)*levels+(40,)+(r, c), deepest level outermost) and the output blocks into the blocked result, so no
# stacked copies are billed.  The circuits compute 8*C; the exact 1/8 is applied once to the blocked left operand,
# out of place (with no levels the blocked operand is a view of the caller's array).

def _kz_graph(graph, inputs, outputs):
    """Evaluate a circuit node by node; output nodes are written into `outputs` with out=."""
    slot = {node: j for (j, node) in enumerate(graph[1])}
    values = {-i - 1: value for (i, value) in enumerate(inputs)}
    for (step, (op, scalar, ids, aa, bb, a_shared, b_shared)) in enumerate(graph[0]):
        operation = fnp.add if op == 'add' else fnp.subtract
        for (node, a, b) in zip(ids, aa, bb):
            j = slot.get(node)
            values[node] = operation(values[a], values[b], out=None if j is None else outputs[j])
        for node in graph[2][step]:
            values.pop(node, None)

def _kz_blocked_top(matrix, schemes, right, rows, columns):
    """prefix+(R, C) -> prefix+(4,)*levels+(rows, columns)+(r, c): a rows x columns top split over the blocked levels."""
    levels = len(schemes)
    prefix = matrix.shape[:-2]
    p = len(prefix)
    r, c = matrix.shape[-2] // (rows << levels), matrix.shape[-1] // (columns << levels)
    split = fnp.reshape(matrix, prefix + (rows,) + (2,) * levels + (r,) + (columns,) + (2,) * levels + (c,))
    bits = p + levels + 3
    if right:
        split = split[(slice(None),) * bits + (slice(None, None, -1),) * levels]
    pairs = ()
    for level in range(levels - 1, -1, -1):
        row, column = p + 1 + level, bits + level
        pairs += (row, column) if (schemes[level] == 'w') == right else (column, row)
    axes = tuple(range(p)) + pairs + (p, p + levels + 2, p + levels + 1, p + 2 * levels + 3)
    return fnp.reshape(fnp.transpose(split, axes), prefix + (4,) * levels + (rows, columns) + (r, c))

def _kz_unblocked_top(blocked, levels):
    """prefix+(4,)*levels+(rows, columns)+(r, c), deepest level first with level index 2c+r -> prefix+(R, C)."""
    prefix = blocked.shape[:-(levels + 4)]
    p = len(prefix)
    rows, columns, r, c = blocked.shape[-4:]
    split = fnp.reshape(blocked, prefix + (2,) * (2 * levels) + (rows, columns, r, c))
    top = p + 2 * levels
    axes = (tuple(range(p)) + (top,) + tuple(p + 2 * (levels - 1 - l) + 1 for l in range(levels)) + (top + 2,) +
            (top + 1,) + tuple(p + 2 * (levels - 1 - l) for l in range(levels)) + (top + 3,))
    return fnp.reshape(fnp.transpose(split, axes), prefix + (rows * (r << levels), columns * (c << levels)))

def _kz_split_top(blocked, axis, rows, columns):
    """The rows*columns top blocks (row-major) of a blocked array whose (rows, columns) axes start at `axis`."""
    return tuple(piece for band in fnp.unstack(blocked, axis=axis) for piece in fnp.unstack(band, axis=axis))

def _kz_r40_matmul(left, right, kind, schemes):
    """Rank-40 top of shape kind=(rm, rk, rn) over len(schemes) in-place levels; dims divisible by kind*2**levels."""
    rm, rk, rn = kind
    levels = len(schemes)
    m, k, n = left.shape[-2] // (rm << levels), left.shape[-1] // (rk << levels), right.shape[-1] // (rn << levels)
    graph_left, graph_right, graph_out = _KP_R40_GRAPHS[kind]
    left_prefix, right_prefix = left.shape[:-2], right.shape[:-2]
    lp, rp = len(left_prefix), len(right_prefix)
    blocked_left = _kz_blocked_top(left, schemes, False, rm, rk)
    blocked_right = _kz_blocked_top(right, schemes, True, rk, rn)
    blocked_left = fnp.multiply(blocked_left, 0.125)
    _kz_basis(blocked_left, lp, schemes, 0, False)
    _kz_basis(blocked_right, rp, schemes, 0, True)
    encoded_left = fnp.empty(left_prefix + (7,) * levels + (40,) + (m, k), dtype=left.dtype)
    encoded_right = fnp.empty(right_prefix + (7,) * levels + (40,) + (k, n), dtype=right.dtype)
    left_regions, right_regions = _kz_regions(schemes, _KZ_LEFT, -1), _kz_regions(schemes, _KZ_RIGHT, -1)
    _kz_graph(graph_left, _kz_split_top(blocked_left, lp + levels, rm, rk),
              fnp.unstack(encoded_left[(slice(None),) * lp + left_regions] if levels else encoded_left, axis=lp + levels))
    _kz_graph(graph_right, _kz_split_top(blocked_right, rp + levels, rk, rn),
              fnp.unstack(encoded_right[(slice(None),) * rp + right_regions] if levels else encoded_right, axis=rp + levels))
    _kz_encode_levels(encoded_left, lp, schemes, 0, False)
    _kz_encode_levels(encoded_right, rp, schemes, 0, True)
    products = fnp.matmul(encoded_left, encoded_right)
    base = len(products.shape) - levels - 3
    _kz_decode(products, base, schemes)
    result = fnp.empty(products.shape[:base] + (4,) * levels + (rm, rn) + (m, n), dtype=products.dtype)
    out_regions = _kz_regions(schemes, _KZ_OUT, -1)
    _kz_graph(graph_out, fnp.unstack(products[(slice(None),) * base + out_regions] if levels else products, axis=base + levels),
              _kz_split_top(result, base + levels, rm, rn))
    _kz_output(result, base, schemes, 0)
    return _kz_unblocked_top(result, levels)


def _kz_r40_route(m, k, n):
    """Rank-40 class rules over five in-place kz levels: <3,3,6> for wide roots (kz_r40_wide: 1 = k in {864, 928},
    2 = k in {768, 832, 864, 896, 928}; m >= 832, n >= 8*max(m, k)) and <3,6,3> for long-inner roots (kz_r40_tall:
    k >= ratio*max(m, n), m >= 768, n >= 864; at most kz_r40_tall_max per predict); every other root keeps the kz dispatch."""
    wide = int(KERNEL_CONFIG.get('kz_r40_wide', 0))
    if wide and n >= 8 * max(m, k) and (m >= 832) and (k in ((864, 928) if wide == 1 else (768, 832, 864, 896, 928))):
        cap = int(KERNEL_CONFIG.get('kz_r40_wide_max', 0))
        if cap <= 0 or _KZ_R40_TAKEN[1] < cap:
            _KZ_R40_TAKEN[1] += 1
            return ((3, 3, 6), 5)
    ratio = int(KERNEL_CONFIG.get('kz_r40_tall', 0))
    if ratio and k >= ratio * max(m, n) and (m >= 768) and (n >= 864):
        # Only where m and n pad to multiples of 3*2**5 within 4 % and the root is not one the kz dispatch takes to
        # depth 7: elsewhere the <3,6,3> top costs FLOPs or saves < 1 M FLOPs per added counted op.
        deep = int(KERNEL_CONFIG.get('kz_deep_work', 0))
        deep7 = deep > 0 and not any((v % 128 for v in (m, k, n))) and min(m, k, n) >= 896 and m * k * n >= deep
        if -(-m // 96) * 96 * (-(-n // 96) * 96) * 25 <= m * n * 26 and (not deep7):
            cap = int(KERNEL_CONFIG.get('kz_r40_tall_max', 0))
            if cap <= 0 or _KZ_R40_TAKEN[0] < cap:
                _KZ_R40_TAKEN[0] += 1
                return ((3, 6, 3), 5)
    return None

def _kz_r40_padded(left, right, kind, depth):
    """Owned padding to kind*2**depth multiples, rank-40 product, crop."""
    (m, k), n = left.shape, right.shape[1]
    pm, pk, pn = ((v + (r << depth) - 1) // (r << depth) * (r << depth) for (v, r) in zip((m, k, n), kind))
    if pm != m or pk != k:
        left = fnp.pad(left, [(0, pm - m), (0, pk - k)], mode='constant')
    if pk != k or pn != n:
        right = fnp.pad(right, [(0, pk - k), (0, pn - n)], mode='constant')
    result = _kz_r40_matmul(left, right, kind, _kz_schemes(depth, pm * pk * pn, 0))
    return result if (pm, pn) == (m, n) else result[:m, :n]

def _fast_matmul(left, right, leg_transport=False):
    if len(left.shape) == 2 and len(right.shape) == 2 and (str(left.dtype) == 'float32') and (str(right.dtype) == 'float32'):
        route = _kz_r40_route(int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
        if route is not None:
            return _kz_r40_padded(left, right, route[0], route[1])
    return _k3_fast_matmul(left, right, leg_transport=leg_transport)
