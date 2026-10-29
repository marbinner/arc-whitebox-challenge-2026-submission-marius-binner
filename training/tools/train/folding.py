"""f3 corrector folding and exact saved-tensor offload, retained from m2train."""
import math
import torch

DEPTH, WIDTH = 16, 1024
NX, NR = 10, 10

RAW_TO_X = [0, 1, 2, 3, 4, 5, 6, 8, 9]

SCALE = [1 / 3, 2.0, 2.5, 1.0, 1.0, 10.0, 10.0, 5.0, -3.0]

SHIFT = [0.0, -1.0, -0.5, -1.0, -1.0, 0.0, 0.0, -10.0, 3.0]

ROOT = math.sqrt(WIDTH)

RC = [1.0, ROOT, 1 / ROOT, 1.0, 1.0, 1.0, 1 / ROOT, 1 / ROOT, 1.0, float(WIDTH)]

GELU_C = 0.7978845608028654

def layer_weights(th, l, depth=DEPTH):
    x = l / max(depth - 1, 1)
    return {k: v[0] + x * v[1] + x * x * v[2] for k, v in th.items()}

def fold_corrector(theta, row_norm, xf_norm):
    """Per-layer folded corrector arrays (float32 stacks over layers), the exact torch twin of port_lean.fold (keeps the graph)."""
    th = {k: v.double() for k, v in theta.items()}
    rm, rs = (t.double() for t in row_norm)
    nb = sum(1 for k in th if k.startswith('Wf'))
    out = {}
    for l in range(DEPTH):
        w = layer_weights(th, l)
        Win, bin_ = w['Win'], w['bin']
        nm = w['Wm'].shape[1]
        Wx = torch.stack([SCALE[j] * Win[RAW_TO_X[j]] for j in range(len(RAW_TO_X))])
        bin_ = bin_ + sum(SHIFT[j] * Win[RAW_TO_X[j]] for j in range(len(RAW_TO_X))) + (l / max(DEPTH - 1, 1)) * Win[7]
        Wr = torch.stack([RC[k] / rs[l, k] * Win[NX + k] for k in range(NR)])
        b_rows = -sum(rm[l, k] / rs[l, k] * Win[NX + k] for k in range(NR))
        Wmi = 0.5 * Win[NX + NR:NX + NR + nm]
        out.setdefault('Wx', []).append(Wx)
        out.setdefault('Wxr', []).append(torch.cat([Wx, Wr]))
        out.setdefault('Wxm', []).append(torch.cat([Wx, Wmi]))
        out.setdefault('Wxrm', []).append(torch.cat([Wx, Wr, Wmi]))
        b = bin_ + (0.0 if l == DEPTH - 1 else b_rows)
        for blk in range(nb):
            out.setdefault(f'Wc{blk}', []).append(w[f'Wc{blk}'])
            out.setdefault(f'Wf{blk}', []).append(GELU_C * w[f'Wf{blk}'])
            out.setdefault(f'bf{blk}', []).append(GELU_C * w[f'bf{blk}'])
            out.setdefault(f'Wg{blk}', []).append(0.5 / GELU_C * w[f'Wg{blk}'])
            out.setdefault(f'bg{blk}', []).append(w[f'bg{blk}'])
        out.setdefault('Wom', []).append(torch.cat([w['Wout'], w['Wm']], 1))
        out.setdefault('bom', []).append(torch.cat([w['bout'], w['bm']]))
        out.setdefault('ABC', []).append(torch.cat([2.0 * w['A'], w['B'] * (ROOT / 2), w['C3'] * (WIDTH / 11.0)], 1))
        out.setdefault('Gz', []).append(0.5 * w['Gz'])
        out.setdefault('bz', []).append(0.5 * w['bz'])
        if xf_norm is not None:
            xm, xs = (t.double() for t in xf_norm); nxf = xm.shape[-1]; o = NX + NR + nm + (w['Q'].shape[1] if 'Q' in w else 0)
            XC = [10.0, 10.0, 1.0, 1.0, 1.0 / ROOT, 10.0][:nxf]
            Wf_ = Win[o:o + nxf]
            out.setdefault('Wxf', []).append(torch.stack([XC[k] / xs[l, k] * Wf_[k] for k in range(nxf)]))
            b = b - sum(xm[l, k] / xs[l, k] * Wf_[k] for k in range(nxf))
        out.setdefault('bin', []).append(b)
    return {k: torch.stack(v).float() for k, v in out.items()}

def _dense_saved_view(tensor):
    """Metadata-only test for non-overlapping dense storage, including transposes."""
    expected = 1
    for stride, size in sorted((stride, size) for stride, size in zip(tensor.stride(), tensor.size()) if size > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def offload_saved(min_mb):
    """saved_tensors_hooks: CUDA tensors saved for backward of ≥ min_mb MB are copied to pinned host memory (same stream → ordered),
    restored on unpack. Exact (bitwise) copies, so loss and gradients are identical to the in-GPU graph."""
    thresh = int(min_mb * 2 ** 20)

    def pack(t):
        if t.is_cuda and t.numel() * t.element_size() >= thresh and 0 not in t.stride():   # expanded (stride-0) views stay on GPU
            # A dense transpose can travel in its own layout. Keeping that layout on
            # the host lets to(device) restore it directly, rather than allocating a
            # contiguous GPU copy followed by a second strided GPU copy.
            c = (torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, pin_memory=True)
                 if _dense_saved_view(t) else torch.empty(t.size(), dtype=t.dtype, pin_memory=True))
            c.copy_(t, non_blocking=True)                                  # dense copy over PCIe (fast)
            return ('cpu', t.device, c, None if t.is_contiguous() else t.stride())
        return t

    def unpack(p):
        if isinstance(p, tuple) and len(p) == 4 and p[0] == 'cpu':
            g = p[2].to(p[1], non_blocking=True)
            if p[3] is None or g.stride() == p[3]:
                return g
            s = torch.empty_strided(g.size(), p[3], dtype=g.dtype, device=g.device)   # restore the saved strides: backward kernels
            s.copy_(g)                                                             # see the in-GPU layout (bit-exact gradients)
            return s
        return p
    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
