"""Separate model, trainer recipe and label-corpus continuation identities."""
import hashlib
import inspect
import json
from pathlib import Path


def recipe_identity(config):
    root = Path(__file__).resolve().parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.glob('*.py'))}
    for name in ('../data.py','../guards.py','../../base/torch_twin.py'):
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    encoded = json.dumps(dict(files=files, config=config), sort_keys=True, allow_nan=False).encode()
    return dict(schema='whest-recipe-identity-v1', sha256=hashlib.sha256(encoded).hexdigest(), files=files)


def loader_identity(loader):
    """Bind code and bound data; opaque loaders must declare their corpus identity.

    __whest_data_identity__ is a JSON value supplied by custom research loaders
    backed by files or other external state. It must identify those actual bytes.
    """
    def bound(value):
        if value is None or type(value) in (bool, int, float, str):
            return [type(value).__name__, value]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [bound(item) for item in value]]
        if isinstance(value, dict):
            return ['dict', sorted([[bound(key), bound(item)] for key,item in value.items()], key=repr)]
        import numpy as np
        if isinstance(value, np.ndarray) and value.dtype.kind != 'O':
            return ['ndarray', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes(order='C')).hexdigest()]
        import torch
        if isinstance(value, torch.Tensor):
            array=value.detach().cpu().contiguous().numpy()
            return ['tensor', bound(array)]
        if isinstance(value, np.generic):
            return ['numpy-scalar', str(value.dtype), value.tobytes().hex()]
        raise ValueError('Opaque custom loader state requires __whest_data_identity__')
    declared=getattr(loader,'__whest_data_identity__',None)
    if declared is None:
        try:
            source=inspect.getsource(loader)
            closure={name:bound(cell.cell_contents) for name,cell in
                     zip(loader.__code__.co_freevars,loader.__closure__ or ())}
            payload=dict(source=source,defaults=bound(loader.__defaults__),
                         kwdefaults=bound(loader.__kwdefaults__),closure=closure)
        except (OSError,TypeError,AttributeError) as error:
            raise ValueError('Opaque custom loader requires __whest_data_identity__') from error
    else:
        try:
            target=loader if inspect.isfunction(loader) else loader.__call__
            source=inspect.getsource(target)
        except (OSError,TypeError,AttributeError) as error:
            raise ValueError('Custom loader identity also requires inspectable callable source') from error
        payload=dict(source=source,declared=declared)
    encoded=json.dumps(payload,sort_keys=True,allow_nan=False).encode()
    return dict(schema='whest-custom-loader-v2',module=getattr(loader,'__module__',type(loader).__module__),
                name=getattr(loader,'__qualname__',type(loader).__qualname__),
                sha256=hashlib.sha256(encoded).hexdigest())
