"""Fresh compiler namespaces and the public runtime API; no private injection."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import runpy
from types import ModuleType
import uuid

import numpy as np
import torch
from ..guards import RuntimeFingerprint
loaded_identity = RuntimeFingerprint.loaded_identity

from .folding import offload_saved
from .torch_compat import prepare_backend, prepare_source
from .offload_arena import SavedTensorArena
from .packed_checkpoint import wrap_source

BUDGET = 2 ** 41
DEFAULT_BASE = Path(__file__).resolve().parents[2] / 'base'
_loader = runpy.run_path(str(DEFAULT_BASE / 'torch_twin.py'))
load_module = _loader['load_module']
compile_kernel = _loader['compile_kernel']
canonical_device = _loader['canonical_device']


def merge_configuration(original,updates):
    merged=dict(original)
    for key,value in (updates or {}).items():
        if key in ('params','pipeline','corrector') and isinstance(value,dict):
            merged[key]={**merged.get(key,{}),**value}
        else:
            merged[key]=value
    return merged


def literal(value):
    """Keep Python structures and native float64-to-float32 setup semantics."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: literal(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(literal(item) for item in value)
    if isinstance(value, list):
        return [literal(item) for item in value]
    return value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


class RuntimeTwin:
    def __init__(self, base=DEFAULT_BASE, device='cpu', *, config=None, packed_checkpoint=False):
        self.base = Path(base).resolve()
        self.device = canonical_device(device)
        self.offload_arena = None
        if type(packed_checkpoint) is not bool: raise ValueError('packed_checkpoint must be boolean')
        self.packed_checkpoint = packed_checkpoint
        config_path=self.base/'RUNTIME_CONFIG.json'
        existing_config=json.loads(config_path.read_text()) if config_path.exists() else {}
        self.config=merge_configuration(existing_config,config)
        self.builder = load_module(self.base / 'build.py')
        self.backend = load_module(self.base / 'torch_api.py')
        self.backend.set_device(self.device)
        prepare_backend(self.backend)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source = prepare_source(self.builder.source('torch'))
        if packed_checkpoint: source = wrap_source(source)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        def metadata(value):
            if isinstance(value,dict): return {str(key):metadata(item) for key,item in value.items()}
            if isinstance(value,(list,tuple)): return [metadata(item) for item in value]
            return literal(value)
        config_bytes=json.dumps(metadata(self.config),sort_keys=True,allow_nan=False).encode()
        self.identity=dict(source_sha256=self.source_sha256,
                           schema='whest-model-identity-v2',
                           backend_sha256=hashlib.sha256((self.base/'torch_api.py').read_bytes()).hexdigest(),
                           training_compat_sha256=hashlib.sha256(Path(__file__).with_name('torch_compat.py').read_bytes()).hexdigest(),
                           dependencies=loaded_identity(),
                           config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                           assets_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in sorted(self.base.glob('*.npz'))})
        provenance=self.base/'TRAINING-PROVENANCE.json'
        self.identity['training_provenance_sha256']=hashlib.sha256(provenance.read_bytes()).hexdigest() if provenance.exists() else None
        kernel = compile_kernel(source, self.backend)
        self.source_config = literal(getattr(kernel, 'runtime_configuration', lambda path: existing_config)(self.base))
        self.config = merge_configuration(self.source_config, config)
        config_bytes=json.dumps(metadata(self.config),sort_keys=True,allow_nan=False).encode()
        self.identity['config_sha256']=hashlib.sha256(config_bytes).hexdigest()
        self.identity['numerical_settings']=dict(device=str(self.device),dtype=str(torch.get_default_dtype()),
                                                threads=torch.get_num_threads(),tf32=False)
        self.model = kernel.Estimator()
        self.mlp_type = kernel.MLP
        self.model.setup(kernel.SetupContext(width=1024, depth=16, seed=0,
                         flop_budget=BUDGET, submission_dir=str(self.base)))
        if config:
            self.model.update_runtime(**config)
        self.schema = self.model.runtime_schema()
        self.source_params={key:literal(spec['default']) for key,spec in self.schema.get('parameters',{}).items()}
        self.source_params.update(self.source_config.get('params',{}))
        if self.schema['version'] != 1:
            raise ValueError('This trainer requires runtime API v1')
        # The normalization function is a public registry function.
        self.normalize_parameter = kernel.normalize_parameter
        self.validate_parameter = kernel.validate_parameter
        self.expand_scalar_parameter = kernel.expand_scalar_parameter
        self.derivations = self.model.setup_derivations()

    def forward(self, weights, request=None, *, offload_mb=0, offload_arena_gb=0, seed=0):
        # A fresh setup graph is mandatory even when the request is unchanged.
        self.backend.set_device(self.device)
        self.model.update_runtime(**(request or {}))
        w = [item.to(self.device, torch.float32) for item in weights]
        mlp = self.mlp_type(width=int(w[0].shape[0]), depth=len(w), weights=w, seed=seed)
        eligible = offload_mb > 0 and torch.is_grad_enabled() and self.device.type == 'cuda'
        if eligible and offload_arena_gb > 0:
            capacity = int(offload_arena_gb * 2**30)
            if self.offload_arena is None:
                self.offload_arena = SavedTensorArena(capacity)
            elif self.offload_arena.capacity_bytes != capacity:
                raise ValueError('An existing offload arena cannot change capacity')
            offload = self.offload_arena.hooks(offload_mb)
        else:
            offload = offload_saved(offload_mb) if eligible else nullcontext()
        with offload:
            return self.model.predict(mlp, BUDGET)

    def runtime_assets(self):
        return self.model.runtime_assets()


class NativeRuntime:
    """Metered functional native predictions for export validation/evaluation."""
    def __init__(self, base=DEFAULT_BASE):
        import flopscope as flops
        from whestbench.sdk import SetupContext
        self.flops = flops
        self.base = Path(base).resolve()
        if (self.base / 'build.py').exists():
            builder = load_module(self.base / 'build.py')
            source = builder.source()
        else:
            source = (self.base / 'estimator.py').read_text()
        kernel = ModuleType('whest_train_native_' + uuid.uuid4().hex)
        exec(compile(source, '<whest-runtime-native>', 'exec'), kernel.__dict__)
        self.model = kernel.Estimator()
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        with flops.budget(BUDGET, quiet=True):
            self.model.setup(SetupContext(width=1024, depth=16, seed=0,
                        flop_budget=BUDGET, api_version='v2-phase2',
                        submission_dir=str(self.base)))

    def forward(self, weights, request=None, *, seed=0):
        import flopscope.numpy as fnp
        from whestbench.domain import MLP
        with self.flops.budget(BUDGET, quiet=True):
            if request:
                self.model.update_runtime(**request)
            mlp = MLP(width=int(weights.shape[1]), depth=len(weights), seed=seed,
                      weights=[fnp.asarray(item.detach().cpu().numpy(), dtype=fnp.float32)
                               for item in weights])
        with self.flops.budget(BUDGET, quiet=True) as budget:
            result = self.model.predict(mlp, BUDGET)
            values = np.asarray(result).copy()
        return values,dict(flops=int(budget.flops_used),counted_ops=len(budget.op_log))
