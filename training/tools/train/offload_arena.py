"""Bound pinned-memory retention across different saved-tensor shapes."""
import weakref
import torch

from .folding import _dense_saved_view


class _Lease:
    def __init__(self, arena):
        self.arena = arena
        self.cursor = 0
        self.live_spans = 0
        self.streams = arena._streams


class _Span:
    def __init__(self, lease):
        self.lease = lease
        lease.live_spans += 1

    def __del__(self):
        self.lease.live_spans -= 1


class SavedTensorArena:
    """A single pinned byte buffer reused only after all saved spans are released.

    Accumulation works when each network is backpropagated before the next.
    A second live graph is rejected, including retained graphs, rather than
    overwriting its saved values. Transfers keep the original CUDA stream order.
    """
    def __init__(self, capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError('Saved-tensor arena capacity must be positive integer bytes')
        self.capacity_bytes = capacity_bytes
        self.buffer = torch.empty(capacity_bytes, dtype=torch.uint8, pin_memory=True)
        self._active = None
        # Streams survive lease collection: async transfers can still be in
        # flight after Python releases the final saved-span/hook references.
        self._streams = {}
        self.peak_used_bytes = 0
        self.graphs = 0
        self.saved_spans = 0

    def hooks(self, min_mb):
        previous = self._active() if self._active is not None else None
        # A completed graph can retain the pack callable without retaining any
        # saved host values. Only live span tokens protect the arena's bytes.
        if previous is not None:
            if previous.live_spans:
                raise RuntimeError('Saved-tensor arena still has a live graph; finish backward before reuse')
        for stream in self._streams.values():
            current = torch.cuda.current_stream(stream.device)
            if current != stream:
                current.wait_stream(stream)
        lease = _Lease(self)
        self._active = weakref.ref(lease)
        self.graphs += 1
        threshold = int(min_mb * 2**20)

        def pack(tensor):
            if not (tensor.is_cuda and tensor.numel() * tensor.element_size() >= threshold
                    and 0 not in tensor.stride()):
                return tensor
            start = (lease.cursor + 63) // 64 * 64
            size = tensor.numel() * tensor.element_size()
            end = start + size
            if end > self.capacity_bytes:
                raise MemoryError('Saved-tensor arena capacity exceeded: '
                                  + str(end) + ' > ' + str(self.capacity_bytes))
            storage = self.buffer[start:end].view(tensor.dtype)
            host = (storage.as_strided(tensor.size(), tensor.stride())
                    if _dense_saved_view(tensor) else storage.view(tensor.size()))
            host.copy_(tensor, non_blocking=True)
            stream = torch.cuda.current_stream(tensor.device)
            lease.streams[tensor.device.index, stream.cuda_stream] = stream
            lease.cursor = end
            self.peak_used_bytes = max(self.peak_used_bytes, end)
            self.saved_spans += 1
            return ('cpu', tensor.device, host,
                    None if tensor.is_contiguous() else tensor.stride(), _Span(lease))

        def unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 5 and packed[0] == 'cpu':
                stream = torch.cuda.current_stream(packed[1])
                packed[4].lease.streams[packed[1].index, stream.cuda_stream] = stream
                restored = packed[2].to(packed[1], non_blocking=True)
                if packed[3] is None or restored.stride() == packed[3]:
                    return restored
                result = torch.empty_strided(restored.size(), packed[3],
                                             dtype=restored.dtype, device=restored.device)
                result.copy_(restored)
                return result
            return packed

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
