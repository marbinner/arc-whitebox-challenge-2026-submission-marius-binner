"""Research backend adapter; excluded from the native submission."""
from types import SimpleNamespace
import numpy as np
import torch

ndarray=torch.Tensor
float32=torch.float32
float64=torch.float64
int64=torch.int64
abs=torch.abs
sqrt=torch.sqrt
exp=torch.exp
log=lambda v:torch.log(v if isinstance(v,torch.Tensor) else torch.as_tensor(v,dtype=torch.float64))
tanh=torch.tanh
outer=torch.outer
eye=torch.eye
arange=torch.arange
zeros_like=torch.zeros_like
ones_like=torch.ones_like
isfinite=torch.isfinite
all=torch.all
not_equal=torch.ne
logical_or=torch.logical_or
logical_not=torch.logical_not
_device=torch.device('cpu')

def set_device(device):
    global _device
    _device=torch.device(device)

def scalar(v):return v if isinstance(v,torch.Tensor) and v.requires_grad and torch.is_grad_enabled() else float(v)
def enabled(v):
    # A zero gain still has its algebraic derivative during a training forward.
    return (isinstance(v,torch.Tensor) and v.requires_grad and torch.is_grad_enabled()) or bool(v != 0)
def asarray(v,dtype=None):
    if isinstance(v,torch.Tensor):
        return v.to(device=_device,dtype=dtype or v.dtype)
    def has_tensor(x):
        return isinstance(x,torch.Tensor) or isinstance(x,(tuple,list)) and any(has_tensor(y) for y in x)
    if isinstance(v,(tuple,list)) and has_tensor(v):
        parts=[asarray(x,dtype) for x in v]
        dt=dtype or parts[0].dtype
        for p in parts:dt=torch.promote_types(dt,p.dtype)
        return torch.stack([p.to(dt) for p in parts])
    return torch.as_tensor(v,dtype=dtype,device=_device)
def asarray_chkfinite(v,dtype=None):
    result=asarray(v,dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('array must not contain infs or NaNs')
    return result
def polyval(coefficients,value):
    coefficients,value=asarray(coefficients),asarray(value)
    result=coefficients[0]
    for coefficient in coefficients[1:]:result=result*value+coefficient
    return result
def zeros(shape,dtype=None):return torch.zeros(shape,dtype=dtype or torch.float64,device=_device)
def ones(shape,dtype=None):return torch.ones(shape,dtype=dtype or torch.float64,device=_device)
def empty(shape,dtype=None):return torch.empty(shape,dtype=dtype or torch.float64,device=_device)
def eye(n,dtype=None):return torch.eye(n,dtype=dtype or torch.float64,device=_device)
def arange(*args,dtype=None):return torch.arange(*args,dtype=dtype,device=_device)
def sum(v,axis=None,keepdims=False,dtype=None):return torch.sum(v,dim=axis,keepdim=keepdims,dtype=dtype)
def cumsum(v,axis=0,dtype=None):return torch.cumsum(v,dim=axis,dtype=dtype)
def flip(v,axis=0):return torch.flip(v,dims=(axis,))
def searchsorted(v,values,side='left'):return torch.searchsorted(v,values,right=side=='right')
def mean(v,axis=None,keepdims=False):return torch.mean(v,dim=axis,keepdim=keepdims)
def min(v,axis=None,keepdims=False):return v.min() if axis is None else torch.amin(v,dim=axis,keepdim=keepdims)
def max(v,axis=None,keepdims=False):return v.max() if axis is None else torch.amax(v,dim=axis,keepdim=keepdims)
def var(v,axis=None,keepdims=False):return torch.var(v,dim=axis,keepdim=keepdims,correction=0)
def std(v,axis=None,dtype=None,keepdims=False,ddof=0):
    v=asarray(v,dtype=dtype)
    return torch.std(v,dim=axis,keepdim=keepdims,correction=ddof)
def maximum(a,b):
    if not isinstance(a,torch.Tensor):a=torch.as_tensor(a,dtype=b.dtype,device=b.device)
    if not isinstance(b,torch.Tensor):b=torch.as_tensor(b,dtype=a.dtype,device=a.device)
    return torch.maximum(a,b)
def where(c,a,b):return torch.where(c,a,b)
def diag(v):return torch.diag(v)
def diagonal(v,offset=0,axis1=0,axis2=1):return torch.diagonal(v,offset,axis1,axis2)
def fill_diagonal(v,d):
    if isinstance(d,torch.Tensor):v.diagonal().copy_(d)
    else:v.diagonal().fill_(d)
def concatenate(v,axis=0,out=None):return torch.cat(v,dim=axis,out=out)
def stack(v,axis=0,out=None):return torch.stack(v,dim=axis,out=out)
def block(v):return torch.cat([torch.cat(row,dim=-1) for row in v],dim=-2)
def einsum(e,*v,optimize=None):
    # Match native NumPy promotion, including float32 with wide integers.
    dtype=getattr(torch,np.result_type(*(dtype_name(value.dtype) for value in v)).name)
    compute_dtype=torch.int64 if dtype==torch.bool else dtype
    if not (compute_dtype.is_floating_point or compute_dtype.is_complex) and v[0].device.type!='cpu':
        # CUDA has no integer einsum kernels; integer contractions are small exact bookkeeping, so run them on CPU.
        return torch.einsum(e,*(value.to(device='cpu',dtype=compute_dtype) for value in v)).to(device=v[0].device,dtype=dtype)
    return torch.einsum(e,*(value.to(dtype=compute_dtype) for value in v)).to(dtype=dtype)
def reshape(v,shape,copy=None):
    result=v.reshape(shape)
    return result.clone() if copy else result
negative=torch.negative
def _live_view(v):
    # Owned buffers can acquire a grad_fn after a destination view is captured.
    # Rebuild that alias against its base's current history before each access.
    if isinstance(v,torch.Tensor) and v._base is not None:
        return v._base.as_strided(v.size(),v.stride(),v.storage_offset())
    return v
def _binary(operation,a,b,out=None,dtype=None):
    a,b=_live_view(a),_live_view(b)
    if dtype is not None or isinstance(a,(tuple,list)):a=asarray(a,dtype=dtype)
    if dtype is not None or isinstance(b,(tuple,list)):b=asarray(b,dtype=dtype)
    result=operation(a,b)
    if out is not None:
        copyto(out,result)
        return out
    return result
def multiply(a,b,out=None,dtype=None):return _binary(torch.multiply,a,b,out,dtype)
matmul=torch.matmul
def copyto(dst,src,casting=None):_live_view(dst).copy_(_live_view(src))
def take(v,indices,axis=0):
    axis=axis if axis>=0 else v.ndim+axis
    indices=torch.as_tensor(indices,dtype=torch.int64,device=v.device)
    result=torch.index_select(v,axis,indices.flatten())
    return result.reshape(v.shape[:axis]+indices.shape+v.shape[axis+1:])
def argsort(v,axis=-1,stable=False):return torch.argsort(v,dim=axis,stable=stable)
def add(a,b,out=None,dtype=None):return _binary(torch.add,a,b,out,dtype)
def subtract(a,b,out=None,dtype=None):return _binary(torch.subtract,a,b,out,dtype)
def transpose(v,*axes):
    if len(axes)==1 and isinstance(axes[0],(tuple,list)):axes=axes[0]
    return v.permute(*axes)
def unstack(v,axis=0):return list(torch.unbind(v,dim=axis))
def split(v,indices,axis=0):
    if isinstance(indices,int):return list(torch.tensor_split(v,indices,dim=axis))
    return list(torch.tensor_split(v,list(indices),dim=axis))
def repeat(v,repeats,axis=None):
    if axis is None:v=v.flatten();axis=0
    return torch.repeat_interleave(v,torch.as_tensor(repeats,device=v.device),dim=axis)
def clip(v,lo,hi):return torch.clamp(v,min=lo,max=hi)
def pad(v,widths,mode='constant'):
    if mode != 'constant':raise ValueError('M1 uses constant zero padding')
    return torch.nn.functional.pad(v,tuple(x for pair in reversed(widths) for x in pair),mode='constant',value=0.0)

class LinearAlgebra:
    qr=staticmethod(torch.linalg.qr)
    solve=staticmethod(torch.linalg.solve)
    cholesky=staticmethod(torch.linalg.cholesky)
    @staticmethod
    def svd(v,full_matrices=False,k=None):
        if v.requires_grad:
            u=LinearAlgebra.svd(v.detach(),full_matrices=False,k=k)[0]
            proj=u.mT@v
            s=proj.norm(dim=-1)
            return u,s,proj/s.clamp_min(1e-30)[...,None]
        u,s,vh=torch.linalg.svd(v,full_matrices=full_matrices)
        return (u,s,vh) if k is None else (u[...,:k],s[...,:k],vh[...,:k,:])

linalg=LinearAlgebra()

class Module:
    @classmethod
    def from_file(cls,path,**overrides):
        obj=cls(**overrides)
        with np.load(path) as z:
            for k in z.files:
                setattr(obj,k,torch.as_tensor(np.array(z[k],copy=True),device=_device))
        return obj

flops=SimpleNamespace(Module=Module,as_symmetric=lambda v,symmetry:v,
                     stats=SimpleNamespace(norm=SimpleNamespace(cdf=lambda v:torch.special.ndtr(v.to(torch.float64)))))
class BaseEstimator:pass
MLP=SimpleNamespace
SetupContext=SimpleNamespace

def dtype_name(dtype):return str(dtype).removeprefix('torch.')
