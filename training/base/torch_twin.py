"""Torch research twin compiled in memory from the native component sources.

The wrapper accepts CPU or CUDA tensors and returns all layer means. CUDA is
supported for the next research stage; M1 validation itself uses CPU only.
"""
import ast
import hashlib
import operator
import importlib.util
from pathlib import Path
from types import ModuleType
import uuid
import torch

HERE = Path(__file__).resolve().parent

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_module(path):
    """Load exact package bytes into a fresh namespace, without import aliases."""
    name = 'whest_isolated_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_kernel(source, backend):
    """Bind generated adapter imports directly; never replace sys.modules."""
    kernel = ModuleType('whest_torch_kernel_' + uuid.uuid4().hex)
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == 'torch_api' for a in node.names):
            for alias in node.names:
                if alias.name != 'torch_api':
                    raise ValueError('Mixed backend import is unsupported')
                kernel.__dict__[alias.asname or alias.name] = backend
        elif isinstance(node, ast.ImportFrom) and node.module == 'torch_api':
            for alias in node.names:
                kernel.__dict__[alias.asname or alias.name] = getattr(backend, alias.name)
        else:
            body.append(node)
    tree.body = body
    exec(compile(tree, '<whest-torch-components>', 'exec'), kernel.__dict__)
    return kernel


def canonical_device(device):
    # CUDA without an index is resolved once, before any runtime is installed.
    return torch.empty(0, device=device).device


def load_kernel(device='cpu'):
    builder = load_module(HERE / 'build.py')
    backend = load_module(HERE / 'torch_api.py')
    backend.set_device(canonical_device(device))
    return compile_kernel(builder.source('torch'), backend)


class Estimator:
    def __init__(self, assets=HERE, device='cpu', params=None,seed=0):
        self.kernel = load_kernel(device)
        self.params = self.kernel.copy_parameter_structure(params) if params is not None else None
        self.seed=operator.index(seed)
        self.assets = Path(assets)
        self.device = canonical_device(device)
        self.estimator = self.kernel.Estimator()
        self.shape = None

    def update_runtime(self, **changes):
        if self.shape is None:
            self._setup((1024, 16), 2**41)
        self.kernel.fnp.set_device(self.device)
        return self.estimator.update_runtime(**changes)

    def _setup(self, shape, budget):
        self.kernel.fnp.set_device(self.device)
        self.estimator.setup(self.kernel.SetupContext(width=shape[0], depth=shape[1], seed=self.seed,
                             flop_budget=budget, submission_dir=str(self.assets)))
        if self.params:
            self.estimator.update_runtime(params=self.params)
        self.shape = (shape, self.device)

    def runtime_assets(self):
        if self.shape is None:
            self.update_runtime()
        return self.estimator.runtime_assets()

    def runtime_schema(self):
        if self.shape is None:
            self.update_runtime()
        return self.estimator.runtime_schema()

    def setup_derivations(self):
        return self.kernel.setup_derivations()

    def predict(self, weights, budget=2**41):
        weights = list(weights)
        if not weights:
            raise ValueError('At least one weight tensor is required')
        shape = (int(weights[0].shape[0]), len(weights))
        device = weights[0].device
        if device != self.device or any(w.device != self.device for w in weights):
            raise ValueError('Twin device is fixed per instance; create another Estimator for a new device')
        if self.shape is None:
            self._setup(shape, budget)
        elif self.shape != (shape, device):
            raise ValueError('Twin shape is fixed after setup; create another Estimator to change shape')
        self.kernel.fnp.set_device(self.device)
        mlp = self.kernel.MLP(width=shape[0], depth=shape[1], weights=weights, seed=self.seed)
        return self.estimator.predict(mlp, budget)


def predict_with_params(weights, params=None, assets=HERE,seed=0):
    return Estimator(assets=assets, device=weights[0].device, params=params,seed=seed).predict(weights)[-1]


predict = predict_with_params


def dependency_hashes(assets=HERE):
    build = load_module(HERE / 'build.py')
    paths = [HERE / 'build.py', HERE / 'torch_twin.py', HERE / 'torch_api.py']
    paths += [HERE / 'whest_est' / (name + '.py') for name in build.COMPONENTS]
    dependencies={str(path.relative_to(HERE)):path for path in paths}
    dependencies.update({'assets/'+name:Path(assets)/name for name in
                         (*build.ASSETS,*getattr(build,'OPTIONAL_ASSETS',()),'RUNTIME_CONFIG.json')})
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for name,path in sorted(dependencies.items())}
