"""Stable native/Torch parameter replacement and setup-dependency refresh API.

Call update_runtime before each training forward. Raw tensors retain their
graphs; all folded tensors and transient caches are rebuilt from current leaves.
Default setup/prediction keeps the launch's expression and operation order.
"""
from __future__ import annotations
from .config import PARAMS, refresh_kernel_config
from .registry import PARAMETER_REGISTRY, ASSET_REGISTRY, expand_scalar_parameter, validate_parameter, normalize_parameter, optional_asset_registry, copy_parameter_structure, validate_k4_schedules
from .state import Runtime, flops, fnp
from .corrector import CORRECTOR_CONFIG, configure_corrector, derive_corrector, derive_joint, derive_v2e, corrector_input_weights, prepare_corrector_inputs, load_v2e, RegisteredWeights
from .features import FEATURE_REGISTRY, DEFAULT_FEATURES, feature_width, feature_layout
from .moments import setup_moment_coefficients
from .stages import compile_pipeline, validate_pipeline
RUNTIME_API_VERSION = 1
RUNTIME_CONFIG = {}
SETUP_DERIVATIONS = {'corrector': {'params': ('r8_output_scale',), 'assets': ('corrector', 'fe31_inputs'), 'provides': ('corrector_layers', 'feature_layers', 'learned.corr', 'learned.corr_in')}, 'v2e': {'params': ('k21_slot', 'k21_slot_v2e_lambda', 'k21_slot_v2e_gates'), 'assets': ('k21_v2e',), 'provides': ('v2e_model', 'v2e_output', 'learned.v2e')}, 'joint': {'params': ('joint_amplitude',), 'assets': ('joint',), 'provides': ('learned.joint',)}, 'moments': {'params': ('heavy_scale', 'edgeworth_34', 't3_k3_k4_gain', 'k4_off_scale'), 'assets': (), 'provides': ('coefficients.heavy', 'coefficients.ew34', 'coefficients.t3')}, 'kernel': {'params': ('*',), 'assets': (), 'provides': ('KERNEL_CONFIG',)}, 'prediction_state': {'params': ('*',), 'assets': ('hadamard',), 'provides': ('initialize_state',), 'when': 'each predict'}}
JOINT_MEMBER_NAMES = {'birth.bias': 'birth__bias', 'birth.weight': 'birth__weight', 'geometry_weight': 'geometry_weight', 'head.bias': 'head__bias', 'head.weight': 'head__weight', 'layer_bias': 'layer_bias', 'head_scale': 'head_scale'}

def setup_derivations():
    return copy_parameter_structure(SETUP_DERIVATIONS)

def runtime_configuration(directory):
    """Optional exported API request, read once during setup."""
    import json
    from pathlib import Path
    config = copy_parameter_structure(RUNTIME_CONFIG)
    path = Path(str(directory)) / 'RUNTIME_CONFIG.json'
    if path.exists():
        config.update(json.loads(path.read_text()))
    return config

def runtime_schema(runtime=None):
    features = RUNTIME_CONFIG.get('corrector', {}).get('features', CORRECTOR_CONFIG['features']) if runtime is None else runtime.corrector_config['features']
    optional = optional_asset_registry(features, FEATURE_REGISTRY)
    feature_schema = {name: {**spec, 'function': spec['function'].__name__} for (name, spec) in FEATURE_REGISTRY.items()}
    schema = dict(version=RUNTIME_API_VERSION, parameters=PARAMETER_REGISTRY, assets={**ASSET_REGISTRY, **optional}, setup_derivations=setup_derivations(), features=feature_schema, feature_layout=feature_layout(features))
    return copy_parameter_structure(schema)

def record_runtime(runtime, width, depth, directory):
    (runtime.width, runtime.depth, runtime.directory) = (int(width), int(depth), directory)
    runtime.params = copy_parameter_structure(PARAMS)
    runtime.per_layer = {}
    runtime.layer_setups = {}
    runtime.linear_model = getattr(runtime, 'linear_model', None)

def runtime_assets(runtime):
    groups = {}
    if 'corr_stacked' in runtime.learned:
        groups['corrector'] = dict(runtime.learned['corr_stacked'])
    if 'fe31_wnew' in runtime.learned:
        groups['fe31_inputs'] = dict(Wnew=runtime.learned['fe31_wnew'], mu=runtime.learned['fe31_mu'], sd=runtime.learned['fe31_sd'])
    if runtime.v2e_model is not None:
        groups['k21_v2e'] = {key: getattr(runtime.v2e_model, key) for key in ('W0', 'b0', 'W1', 'b1', 'Wout', 'bout', 'mean', 'std')}
    if runtime.joint_weights:
        groups['joint'] = {JOINT_MEMBER_NAMES[key]: value for (key, value) in runtime.joint_weights.items()}
    if runtime.projection is not None:
        groups['hadamard'] = dict(production=runtime.projection)
    if runtime.linear_model is not None:
        groups['linear'] = dict(runtime.linear_model)
    if runtime.feature_weights:
        groups['feature_weights'] = dict(runtime.feature_weights)
    return groups

def validate_runtime_value(name, value, schema):
    spec = schema[name]
    if not hasattr(value, 'shape'):
        value = fnp.asarray(value, dtype=fnp.float32 if spec['dtype'] == 'float32' else fnp.float64)
    if tuple(value.shape) != tuple(spec['shape']) or str(value.dtype).removeprefix('torch.') != spec['dtype']:
        raise ValueError('Asset tensor shape/dtype mismatch: ' + name)
    if not bool(fnp.all(fnp.isfinite(value))):
        raise ValueError('Nonfinite runtime tensor: ' + name)
    if name.endswith(('.sd', '.std')) and (not bool(fnp.all(value > 0))):
        raise ValueError('Normalization standard deviation must be positive: ' + name)
    return value

def validate_parameter_tensors(name, value):
    if isinstance(value, dict):
        for v in value.values():
            validate_parameter_tensors(name, v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            validate_parameter_tensors(name, v)
    elif hasattr(value, 'shape'):
        if getattr(value, 'requires_grad', False):
            if name == 'k4_d40o_scale':
                raise ValueError('K4 storage routing requires a fixed literal: ' + name)
            if name == 'k4_star_scale' and bool(fnp.all(value == 0)):
                raise ValueError('Trainable K4 normalization is singular at zero: ' + name)
        if not bool(fnp.all(fnp.isfinite(value))):
            raise ValueError('Nonfinite parameter tensor: ' + name)
        bounds = PARAMETER_REGISTRY[name].get('bounds')
        if bounds and (not bool(fnp.all((value >= bounds[0]) & (value <= bounds[1])))):
            raise ValueError('Parameter tensor out of bounds: ' + name)

def install_runtime_assets(runtime, updates):
    if 'k21_v2e' in updates and runtime.v2e_model is None:
        from pathlib import Path
        runtime.v2e_model = RegisteredWeights.from_file(str(Path(runtime.directory) / 'k21_v2e.npz'), group='k21_v2e')
        runtime.v2e_output = (runtime.v2e_model.Wout, runtime.v2e_model.bout)
    assets = runtime_assets(runtime)
    for (group, members) in updates.items():
        assets.setdefault(group, {}).update(members)
    if 'corrector' in assets:
        runtime.learned['corr_stacked'] = assets['corrector']
    if 'fe31_inputs' in assets:
        f = assets['fe31_inputs']
        runtime.learned.update(fe31_wnew=f['Wnew'], fe31_mu=f['mu'], fe31_sd=f['sd'])
    if 'joint' in assets:
        runtime.joint_weights = {key: assets['joint'][member] for (key, member) in JOINT_MEMBER_NAMES.items()}
    if 'k21_v2e' in assets:
        if runtime.v2e_model is None:
            runtime.v2e_model = RegisteredWeights('k21_v2e')
        for (key, value) in assets['k21_v2e'].items():
            setattr(runtime.v2e_model, key, value)
        runtime.v2e_output = (runtime.v2e_model.Wout, runtime.v2e_model.bout)
    if 'hadamard' in assets:
        runtime.projection = assets['hadamard']['production']
    if 'linear' in assets:
        runtime.linear_model = assets['linear']
    if 'feature_weights' in assets:
        runtime.feature_weights = assets['feature_weights']

def refresh_runtime(runtime, groups=None):
    """Rebuild named setup products from raw leaves; default refreshes all."""
    groups = set(('corrector', 'v2e', 'joint', 'moments', 'kernel') if groups is None else groups)
    if groups - set(SETUP_DERIVATIONS):
        raise ValueError('Unknown setup derivation')
    PARAMS.update(runtime.params)
    refresh_kernel_config()
    if runtime.width != 1024 or runtime.depth != 16:
        return sorted(groups)
    if 'corrector' in groups:
        stacked = runtime.learned['corr_stacked']
        runtime.corrector_layers = [{key: value[layer] for (key, value) in stacked.items()} for layer in range(16)]
        if 'fe31_wnew' in runtime.learned:
            (W, mu, sd) = (runtime.learned[key] for key in ('fe31_wnew', 'fe31_mu', 'fe31_sd'))
            runtime.feature_layers = [(W[layer], mu[layer], sd[layer]) for layer in range(16)]
        runtime.learned['corr'] = derive_corrector(runtime)
        runtime.learned['corr_in'] = {}
        prepare_corrector_inputs(runtime, runtime.depth)
        for layer in range(runtime.depth):
            corrector_input_weights(runtime, layer, (layer + 1 < runtime.depth, layer > 0, runtime.feature_layers is not None and layer > 0, False))
    if 'v2e' in groups:
        if int(PARAMS['k21_slot']):
            if runtime.v2e_model is None:
                load_v2e(runtime, runtime.directory, runtime.width, runtime.depth)
            gates = tuple((float(value) * float(PARAMS['k21_slot_v2e_lambda']) for value in PARAMS['k21_slot_v2e_gates']))
            runtime.learned['v2e'] = derive_v2e(runtime, runtime.width, gates)
        else:
            runtime.learned.pop('v2e', None)
    if 'joint' in groups:
        runtime.learned['joint'] = derive_joint(runtime, runtime.width, runtime.depth)
    if 'moments' in groups:
        setup_moment_coefficients(runtime, runtime.width, runtime.depth)
    runtime.readout.clear()
    return sorted(groups)

def prepare_layer_setups(runtime):
    runtime.layer_setups = {}
    if not runtime.per_layer:
        return
    groups = {name for name, spec in SETUP_DERIVATIONS.items()
              if spec.get('when') != 'each predict'
              and ('*' in spec['params'] or set(spec['params']) & set(runtime.per_layer))}
    for layer in range(16):
        layer_runtime = Runtime()
        layer_runtime.__dict__.update(runtime.__dict__)
        layer_runtime.learned = dict(runtime.learned)
        layer_runtime.coefficients = dict(runtime.coefficients)
        layer_runtime.readout = {}
        layer_runtime.params = dict(runtime.params)
        layer_runtime.params.update({name: values[layer] for (name, values) in runtime.per_layer.items()})
        refresh_runtime(layer_runtime, groups)
        runtime.layer_setups[layer] = layer_runtime
    PARAMS.update(runtime.params)
    refresh_kernel_config()

def update_runtime(runtime, *, params=None, assets=None, per_layer=None, pipeline=None, corrector=None):
    """Validate then atomically install raw values and refresh their dependents."""
    proposed = copy_parameter_structure(runtime.params)
    for (name, value) in (params or {}).items():
        if name not in PARAMETER_REGISTRY:
            raise ValueError('Unknown parameter: ' + name)
        value = normalize_parameter(name, value)
        validate_parameter(name, value)
        validate_parameter_tensors(name, value)
        proposed[name] = copy_parameter_structure(value)
    schedules = runtime.per_layer if per_layer is None else dict(per_layer)
    for (name, values) in schedules.items():
        expand_scalar_parameter(name, values)
        validate_parameter_tensors(name, values)
    validate_k4_schedules(proposed, schedules)
    pipeline_choices = dict(runtime.pipeline_config)
    pipeline_choices.update(pipeline or {})
    validate_pipeline(pipeline_choices)
    corrector_choices = dict(runtime.corrector_config)
    corrector_choices.update(corrector or {})
    probe = Runtime()
    probe.__dict__.update(runtime.__dict__)
    (probe.params, probe.per_layer) = (proposed, copy_parameter_structure(schedules))
    (probe.learned, probe.coefficients, probe.readout) = (dict(runtime.learned), dict(runtime.coefficients), {})
    probe.feature_weights = {name: value for (name, value) in runtime.feature_weights.items() if name in corrector_choices['features']}
    same_features = tuple(corrector_choices['features']) == runtime.corrector_config['features']
    probe.linear_model = dict(runtime.linear_model) if same_features and runtime.linear_model is not None else None
    if runtime.v2e_model is not None:
        probe.v2e_model = type(runtime.v2e_model).__new__(type(runtime.v2e_model))
        probe.v2e_model.__dict__.update(runtime.v2e_model.__dict__)
    configure_corrector(probe, corrector_choices)
    schema = runtime_schema(probe)['assets']
    checked = {}
    for (group, members) in (assets or {}).items():
        if not isinstance(members, dict):
            raise ValueError('Assets must be named tensor dictionaries')
        checked[group] = {}
        for (key, value) in members.items():
            name = group + '.' + key
            if name not in schema:
                raise ValueError('Unknown learned tensor: ' + name)
            checked[group][key] = validate_runtime_value(name, value, schema)
    install_runtime_assets(probe, checked)
    configure_corrector(probe, corrector_choices)
    compile_pipeline(probe, pipeline_choices)
    previous_globals = copy_parameter_structure(PARAMS)
    try:
        refreshed = refresh_runtime(probe)
        prepare_layer_setups(probe)
    except Exception:
        PARAMS.update(previous_globals)
        refresh_kernel_config()
        raise
    runtime.__dict__.update(probe.__dict__)
    return dict(version=RUNTIME_API_VERSION, refreshed=refreshed, changed_params=list(params or {}), changed_assets=list(checked), per_layer=list(runtime.per_layer))

def activate_prediction_runtime(runtime):
    PARAMS.update(runtime.params)
    refresh_kernel_config()
    runtime.readout.clear()

def activate_layer_parameters(state, layer):
    runtime = state.runtime
    if not runtime.per_layer:
        return
    active = runtime.layer_setups[layer]
    PARAMS.update(active.params)
    runtime.coefficients = active.coefficients
    for key in ('corr', 'corr_in', 'v2e', 'joint'):
        if key in active.learned:
            runtime.learned[key] = active.learned[key]
    for key in ('leg_level_gain', 'final_k3_scale', 'final_k4_scale', 'pair_ck3', 't2_k3_scale', 'split_pair_c3_cov', 'cov_adt3', 'k4_star_scale', 'k4_22_dress'):
        setattr(state, key, PARAMS[key])
    state.dress_rate = PARAMS['k3_vertex_dress'] / 6.0
    (state.cov_k3sq, state.cov_k3cross) = (PARAMS['cov_k3sq'] / 6.0, PARAMS['cov_k3cross'] / 6.0)
    state.carrier_terms = [('mu_var', PARAMS['k3_carrier_mu_var'])] if PARAMS['k3_carrier_mu_var'] != 0.0 else []
    shape = PARAMS['t2_k3_shape']
    state.t2_k3_coefficients = fnp.asarray((0.25 * state.t2_k3_scale * shape[0], 0.5 * state.t2_k3_scale * shape[1], 0.25 * state.t2_k3_scale * shape[2]), dtype=fnp.float32)
    state.k4_family_ratios = (PARAMS['k4_d22c_scale'] / state.k4_star_scale, PARAMS['k4_d31c_scale'] / state.k4_star_scale, 1.0 / state.k4_star_scale) if state.k4_star_scale != 0.0 else (0.0, 0.0, 0.0)
    state.cov_weight = PARAMS['k4_cov_weight']
    state.identity_share = 1.0 - state.cov_weight
    state.cov_frobenius_share = state.cov_weight * state.cov_weight
    state.identity_frobenius_share = 1.0 - state.cov_frobenius_share
    if 'k4_cov_weight' in runtime.per_layer and layer > 0:
        state.identity_gain = state.identity_share * fnp.sum(fnp.diag(state.covariance)) / float(state.width)
    state.cov_gain = state.cov_weight

class LinearWeights(flops.Module):

    def __init__(self):
        self.weight = fnp.zeros((), dtype=fnp.float32)
        self.bias = fnp.zeros((), dtype=fnp.float32)

class FeatureWeights(flops.Module):

    def __init__(self, features=None):
        for name in features if features is not None else CORRECTOR_CONFIG['features']:
            if name not in DEFAULT_FEATURES:
                setattr(self, name, fnp.zeros((), dtype=fnp.float32))

def load_optional_corrector_assets(runtime, directory):
    from pathlib import Path
    path = Path(str(directory)) / 'linear.npz'
    if path.exists():
        weights = LinearWeights.from_file(str(path))
        runtime.linear_model = dict(weight=weights.weight, bias=weights.bias)
        configure_corrector(runtime, runtime.corrector_config)
    path = Path(str(directory)) / 'feature_weights.npz'
    if path.exists():
        weights = FeatureWeights.from_file(str(path), features=runtime.corrector_config['features'])
        runtime.feature_weights = {name: getattr(weights, name) for name in runtime.corrector_config['features'] if name not in DEFAULT_FEATURES}
