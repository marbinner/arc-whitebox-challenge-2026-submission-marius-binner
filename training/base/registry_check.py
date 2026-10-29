"""Build-time schema and literal audit. Host only; excluded from submissions."""
import ast
from collections import Counter
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

SCIENTIFIC_MODULES = ('state', 'array_ops', 'selection', 'moments', 'cumulants4',
                      'history', 'cumulants3', 'corrector', 'initialize', 'estimator', 'stages', 'features', 'runtime_api', 'linalg')


def literal_assignment(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return node.value


def metadata_literal(node):
    """Decode static metadata, including the documented nested dict(...) syntax."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        if node.args:
            raise ValueError('Feature metadata dict() only supports keyword entries')
        result = {}
        for keyword in node.keywords:
            value = metadata_literal(keyword.value)
            if keyword.arg is None:
                result.update(value)
            else:
                result[keyword.arg] = value
        return result
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(metadata_literal(value))
            else:
                result[metadata_literal(key)] = metadata_literal(value)
        return result
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [metadata_literal(value) for value in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    return ast.literal_eval(node)


def configured_feature_schema(package):
    """Read literal feature metadata without importing either numerical backend."""
    modules = package / 'whest_est'
    config = ast.literal_eval(literal_assignment(modules / 'corrector.py', 'CORRECTOR_CONFIG'))
    runtime = ast.literal_eval(literal_assignment(modules / 'runtime_api.py', 'RUNTIME_CONFIG'))
    sidecar = package / 'RUNTIME_CONFIG.json'
    if sidecar.exists(): runtime.update(json.loads(sidecar.read_text()))
    config.update(runtime.get('corrector', {}))
    node = literal_assignment(modules / 'features.py', 'FEATURE_REGISTRY')
    feature_entries = list(zip(node.keys, node.values))
    # Source modules may append a public entry using the documented syntax,
    # FEATURE_REGISTRY['name'] = dict(...), without rewriting the initial map.
    for statement in ast.parse((modules / 'features.py').read_text()).body:
        if not isinstance(statement, ast.Assign): continue
        for target in statement.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'FEATURE_REGISTRY':
                feature_entries.append((target.slice, statement.value))
    specs = {}
    for key, value in feature_entries:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict':
            spec = {}
            for keyword in value.keywords:
                if keyword.arg is None: spec.update(metadata_literal(keyword.value))
                elif keyword.arg != 'function': spec[keyword.arg] = metadata_literal(keyword.value)
        elif isinstance(value, ast.Dict):
            spec = {ast.literal_eval(k): metadata_literal(v) for k, v in zip(value.keys, value.values)
                    if ast.literal_eval(k) != 'function'}
        else:
            raise ValueError('Feature registry entries must carry literal metadata')
        specs[ast.literal_eval(key)] = spec
    features = tuple(config['features'])
    if len(set(features)) != len(features) or set(features) - set(specs):
        raise ValueError('Unknown or duplicate configured feature')
    implementations = literal_assignment(modules / 'corrector.py', 'CORRECTOR_IMPLEMENTATIONS')
    known = {ast.literal_eval(key) for key in implementations.keys}
    for statement in ast.parse((modules / 'corrector.py').read_text()).body:
        if not isinstance(statement, ast.Assign): continue
        for target in statement.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'CORRECTOR_IMPLEMENTATIONS':
                known.add(ast.literal_eval(target.slice))
    if config['implementation'] not in known:
        raise ValueError('Unknown configured corrector')
    return features, specs, runtime


def validate_stage_config(package, config):
    """Run the real configuration validator with symbolic numerical functions."""
    tree = ast.parse((package / 'whest_est/stages.py').read_text())
    assignments = {'STAGE_ORDER', 'PIPELINE_CONFIG', 'STAGE_ALIASES', 'STAGE_FIELDS',
                   'STAGE_IMPLEMENTATIONS', 'STAGE_REGISTRY'}
    nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in assignments for t in n.targets)) or
        (isinstance(n, ast.FunctionDef) and n.name in ('stage_contracts', 'validate_pipeline'))]
    implementations = literal_assignment(package / 'whest_est/stages.py', 'STAGE_IMPLEMENTATIONS')
    names = {n.id for n in ast.walk(implementations) if isinstance(n, ast.Name)}
    namespace = {name: SimpleNamespace(__name__=name) for name in names}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<stage-schema>', 'exec'), namespace)
    return namespace['validate_pipeline'](config)


def float_inventory(package):
    found = {}
    for module in SCIENTIFIC_MODULES:
        file = package / 'whest_est' / (module + '.py')
        class Scanner(ast.NodeVisitor):
            scope = 'module'
            def visit_FunctionDef(self, node):
                previous, self.scope = self.scope, node.name
                self.generic_visit(node)
                self.scope = previous
            def visit_Constant(self, node):
                if isinstance(node.value, float):
                    found.setdefault(module + '.' + self.scope, Counter())[repr(node.value)] += 1
        Scanner().visit(ast.parse(file.read_text()))
    return found


def check_registry(package, check_assets=True):
    package = Path(package)
    registry = runpy.run_path(str(package / 'whest_est/registry.py'))
    params_module = dict(registry)
    params_tree = ast.parse((package / 'whest_est/params.py').read_text())
    params_tree.body = [n for n in params_tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    exec(compile(params_tree, '<registry-derived-params>', 'exec'), params_module)
    params, space = params_module['PARAMS'], params_module['PARAM_SPACE']
    entries = registry['PARAMETER_REGISTRY']
    if set(params) != set(space) or set(params) != set(entries):
        raise ValueError('PARAMS/PARAM_SPACE/registry key mismatch')
    for key, value in params.items():
        entry = entries[key]
        required = {'name', 'default', 'shape', 'kind', 'trainable', 'owner', 'doc'}
        if required - set(entry) or not ({'bounds', 'allowed'} & set(entry)):
            raise ValueError('Incomplete registry entry: ' + key)
        if value != entry['default']:
            raise ValueError('Registry/literal default drift: ' + key)
        registry['validate_parameter'](key, value)
        for option in space[key].get('choice', []):
            registry['validate_parameter'](key, option)
    allowed = json.loads((package / 'FLOAT-CONSTANTS.json').read_text())
    found = float_inventory(package)
    for scope, constants in found.items():
        for value, count in constants.items():
            spec = allowed.get(scope, {}).get(value)
            if not spec or not spec.get('reason') or count > spec['count']:
                raise ValueError('Unregistered float literal: ' + scope + ' value=' + value)
    tensors = registry['ASSET_REGISTRY']
    feature_names, feature_specs, runtime = configured_feature_schema(package)
    if not isinstance(runtime, dict) or set(runtime) - {'params', 'per_layer', 'pipeline', 'corrector'}:
        raise ValueError('Export configuration supports params, per_layer, pipeline and corrector')
    validate_stage_config(package, runtime.get('pipeline'))
    for name, value in runtime.get('params', {}).items():
        registry['validate_parameter'](name, registry['normalize_parameter'](name, value))
    for name, values in runtime.get('per_layer', {}).items():
        registry['expand_scalar_parameter'](name, values)
    registry['validate_k4_schedules']({**params, **runtime.get('params', {})}, runtime.get('per_layer', {}))
    optional = registry['optional_asset_registry'](feature_names, feature_specs)
    optional_seen = set()
    if check_assets:
        import numpy as np
        seen = set()
        for file in sorted({entry['asset'] for entry in tensors.values()}):
            with np.load(package / file, allow_pickle=False) as arrays:
                for key in arrays.files:
                    name = file.removesuffix('.npz') + '.' + key
                    if name == 'hadamard.signs':
                        continue  # Accepted legacy input; build discards it.
                    if name not in tensors:
                        raise ValueError('Unregistered asset tensor: ' + name)
                    registry['validate_asset_tensor'](name, arrays[key])
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError('Nonfinite learned asset: ' + name)
                    if name.endswith(('.sd', '.std')) and not (arrays[key] > 0).all():
                        raise ValueError('Normalization standard deviation must be positive: ' + name)
                    seen.add(name)
        if seen != set(tensors):
            raise ValueError('Missing registered learned tensor')
        for file, group in (('linear.npz', 'linear'), ('feature_weights.npz', 'feature_weights')):
            if not (package / file).exists(): continue
            expected = {name for name in optional if name.startswith(group + '.')}
            with np.load(package / file, allow_pickle=False) as arrays:
                supplied = {group + '.' + key for key in arrays.files}
                if supplied != expected: raise ValueError('Optional learned tensor keys mismatch: ' + file)
                for key in arrays.files:
                    name = group + '.' + key
                    value, spec = arrays[key], optional[name]
                    if tuple(value.shape) != tuple(spec['shape']) or str(value.dtype) != spec['dtype']:
                        raise ValueError('Optional learned tensor shape/dtype mismatch: ' + name)
                    if not np.isfinite(value).all(): raise ValueError('Nonfinite learned asset: ' + name)
                    optional_seen.add(name)
    return dict(parameters=len(entries), asset_tensors=len(tensors),
                optional_asset_tensors=len(optional_seen), float_scopes=len(found), passed=True)
