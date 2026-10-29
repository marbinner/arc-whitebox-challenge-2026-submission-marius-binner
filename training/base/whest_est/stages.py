"""Declared layer pipeline, field contracts, and consistent stage alternatives.

Implementations mutate only their explicit state/frame arguments. Production
default dispatch preserves the original fifteen function calls and order.
"""
from __future__ import annotations
from .state import LayerFrame, W01, W31, fnp
from .moments import prepare_layer, relu_moments, finish_moments
from .cumulants3 import transport_cumulants3, calibrate_k3_level, prepare_source_birth, store_source_birth
from .cumulants4 import propagate_cumulants4, store_k4_vertices, update_k4_host
from .corrector import apply_joint_geometry, apply_corrector, correct_post_moments
from .history import apply_birth_diagonal, finish_history

STAGE_FIELDS = {
    'transport': dict(module='moments',
        requires=('state.col_alive_tiers state.covariance state.covariance_selection state.mean state.mlp'
                  ' state.ones_vector state.pair_kink_rows state.row_alive_tiers state.runtime'
                  ' state.terminal_kink_rows state.wick_coefficients state.width state.zeros_vector'
                  ' frame.layer frame.weight').split(),
        provides=('frame.active_rows frame.birth_delta frame.birth_diagonal frame.birth_order'
                  ' frame.birth_slice frame.births_active frame.block_first frame.col_count'
                  ' frame.covariance_pre frame.expand_terminal_births frame.family_channel'
                  ' frame.far_moment_data frame.gaussian frame.is_final frame.mean_pre'
                  ' frame.pair_inverse_order frame.pair_order frame.pair_selected frame.pair_window'
                  ' frame.radial_birth_input frame.row_inverse frame.row_keep frame.row_order frame.row_space'
                  ' frame.star_channel frame.terminal_births frame.terminal_inverse_order frame.terminal_order'
                  ' frame.terminal_selected frame.variance_floor frame.variance_pre frame.variance_squared'
                  ' frame.weight_squared frame.wick state.covariance_selection state.pair_kink_rows').split(),
        nonterminal_only=False),
    'k3_legs': dict(module='cumulants3',
        requires=('state.birth_c2 state.birth_history state.birth_self_c state.carried_gain'
                  ' state.carrier_terms state.leg_dense state.leg_left state.leg_level_gain'
                  ' state.leg_path_right state.leg_q_dense state.leg_right state.leg_tiers state.mlp'
                  ' state.pair_kink_rows state.q_offsets state.row_inverse_prev state.runtime'
                  ' state.source_alive state.source_free_diagonals state.source_free_u'
                  ' state.source_path_factors state.source_remainder_left state.source_remainder_right'
                  ' state.source_singleton_diagonals state.source_transport_alive state.terminal_kink_rows'
                  ' state.width state.zeros_vector frame.active_rows frame.birth_order frame.births_active'
                  ' frame.block_first frame.col_count frame.expand_terminal_births frame.family_channel'
                  ' frame.gaussian frame.is_final frame.layer frame.mean_pre frame.pair_inverse_order'
                  ' frame.pair_order frame.pair_selected frame.pair_window frame.row_inverse frame.row_keep'
                  ' frame.row_space frame.star_channel frame.terminal_births frame.terminal_inverse_order'
                  ' frame.terminal_order frame.terminal_selected frame.variance_floor frame.variance_pre'
                  ' frame.weight frame.wick').split(),
        provides=('frame.birth_diagonal frame.birth_slice frame.block_first frame.family_channel'
                  ' frame.far_moment_data frame.groups frame.k3_21 frame.k3_diagonal frame.pre_transport'
                  ' frame.repeated_parts frame.star_channel frame.suffix_parts frame.transport_alive').split(),
        nonterminal_only=False),
    'k4_host_stars': dict(module='cumulants4',
        requires=('state.cov_gain state.identity_gain state.identity_share state.k4_22_dress state.k4_core'
                  ' state.k4_star_scale frame.covariance_pre frame.is_final frame.layer frame.star_channel'
                  ' frame.variance_pre frame.weight frame.weight_squared').split(),
        provides=('frame.k4_22 frame.k4_diagonal frame.radial_diagonal frame.radial_k4 frame.star_channel').split(),
        nonterminal_only=False),
    'joint_geometry': dict(module='corrector',
        requires=('state.ones_vector state.runtime frame.covariance_pre'
                  ' frame.gaussian frame.k3_21 frame.k3_diagonal frame.k4_22 frame.k4_diagonal frame.layer'
                  ' frame.weight frame.weight_squared frame.wick').split(),
        provides=('frame.geometry_delta frame.geometry_heads frame.has_joint frame.joint_h frame.joint_heads'
                  ' frame.joint_sigma frame.k3_21 frame.k3_diagonal frame.k4_22 frame.k4_diagonal'
                  '').split(),
        nonterminal_only=False),
    'h12_fe30': dict(module='history',
        requires=('state.runtime frame.birth_delta frame.birth_diagonal frame.k4_diagonal frame.layer'
                  ' frame.radial_birth_input frame.radial_k4').split(),
        provides=('frame.birth_delta frame.k4_diagonal frame.radial_birth_input').split(),
        nonterminal_only=False),
    'k3_calibration': dict(module='cumulants3',
        requires=('frame.k3_diagonal frame.layer').split(),
        provides=('frame.k3_diagonal').split(),
        nonterminal_only=False),
    'v2e_r8': dict(module='corrector',
        requires=('state.corrector_carry state.covariance state.k4_core state.mlp state.runtime'
                  ' frame.birth_delta frame.birth_diagonal frame.birth_slice frame.covariance_pre'
                  ' frame.gaussian frame.is_final frame.k3_21 frame.k3_diagonal frame.k4_22 frame.k4_diagonal'
                  ' frame.layer frame.mean_pre frame.radial_birth_input frame.radial_diagonal frame.radial_k4'
                  ' frame.variance_floor frame.weight frame.weight_squared').split(),
        provides=('frame.k3_21 frame.mean_delta frame.radial_birth_input state.corrector_carry').split(),
        nonterminal_only=False),
    'relu_moments': dict(module='moments',
        requires=('state.cov_adt3 state.cov_k3cross state.cov_k3sq state.dress_rate state.final_k3_scale'
                  ' state.final_k4_scale state.pair_ck3 state.power_mix state.rows state.runtime'
                  ' state.split_pair_c3_cov state.t2_k3_coefficients state.t2_k3_scale state.zeros_vector'
                  ' frame.covariance_pre frame.far_moment_data frame.is_final frame.k3_21 frame.k3_diagonal'
                  ' frame.k4_22 frame.k4_diagonal frame.layer frame.mean_delta frame.mean_pre'
                  ' frame.variance_floor frame.variance_squared frame.wick').split(),
        provides=('frame.covariance_offdiag frame.far_moment_data frame.k2_diagonal frame.k2_offdiag'
                  ' frame.k3_out_21 frame.k3_out_diagonal frame.k4_slices frame.offdiag_frobenius'
                  ' frame.ordinary_shared frame.p1 frame.p1_v11 frame.power_rows frame.powers frame.v01'
                  ' frame.v11 frame.w11 frame.w21 frame.w71').split(),
        nonterminal_only=False),
    'source_birth': dict(module='cumulants3',
        requires=('state.carried_gain state.remainder_projection state.runtime state.transport_tiers'
                  ' state.width state.zeros_vector frame.covariance_offdiag frame.k3_21 frame.k3_diagonal'
                  ' frame.k3_out_21 frame.k3_out_diagonal frame.k4_diagonal frame.layer frame.mean_pre'
                  ' frame.p1 frame.p1_v11 frame.pre_transport frame.row_keep frame.row_order frame.v01'
                  ' frame.v11 frame.variance_floor frame.w11 frame.w21 frame.w71 frame.wick').split(),
        provides=('frame.alive frame.alive_count frame.free_diagonal frame.path_repeated frame.remainder_left'
                  ' frame.remainder_right frame.singleton_diagonal frame.transport_alive frame.u_first'
                  ' frame.w01 frame.w31 state.carried_gain').split(),
        nonterminal_only=True),
    'k4_vertices': dict(module='cumulants4',
        requires=('state.k4_family_ratios state.source_k4_birth_ratios state.source_k4_family_vertices state.source_k4_star_vertices'
                  ' frame.alive frame.mean_pre frame.ordinary_shared frame.p1 frame.power_rows'
                  ' frame.variance_floor frame.w01 frame.w11 frame.w21 frame.w31').split(),
        provides=[],
        nonterminal_only=True),
    'source_storage': dict(module='cumulants3',
        requires=('state.birth_c2 state.birth_self_c state.q_offsets state.source_alive'
                  ' state.source_free_diagonals state.source_free_u state.source_k4_family_vertices'
                  ' state.source_k4_star_vertices state.source_path_factors state.source_remainder_left'
                  ' state.source_remainder_right state.source_singleton_diagonals state.source_transport_alive'
                  ' frame.alive frame.free_diagonal frame.mean_pre frame.path_repeated frame.remainder_left'
                  ' frame.remainder_right frame.singleton_diagonal frame.transport_alive frame.u_first'
                  ' frame.variance_floor frame.w11 frame.w21').split(),
        provides=('frame.free_diagonal frame.path_repeated frame.singleton_diagonal frame.u_first').split(),
        nonterminal_only=True),
    'post_corrections': dict(module='corrector',
        requires=('frame.geometry_delta frame.geometry_heads frame.has_joint frame.joint_h frame.joint_heads'
                  ' frame.joint_sigma frame.k2_diagonal frame.k2_offdiag frame.layer frame.p1').split(),
        provides=('frame.k2_diagonal frame.k2_offdiag frame.p1').split(),
        nonterminal_only=True),
    'host_update': dict(module='cumulants4',
        requires=('state.cov_frobenius_share state.cov_weight state.identity_frobenius_share'
                  ' state.identity_share state.width frame.k2_diagonal frame.k4_slices frame.offdiag_frobenius').split(),
        provides=('state.cov_gain state.identity_gain state.k4_core').split(),
        nonterminal_only=True),
    'mean_cov': dict(module='moments',
        requires=('frame.k2_diagonal frame.k2_offdiag frame.mean_delta frame.p1').split(),
        provides=('state.covariance state.mean').split(),
        nonterminal_only=True),
    'history_feed': dict(module='history',
        requires=('state.birth_history state.carried_gain state.feed_active state.leg_left'
                  ' state.leg_path_right state.leg_q_dense state.leg_right state.mean state.q_offsets'
                  ' state.rows state.runtime state.source_path_factors state.source_transport_alive'
                  ' frame.groups frame.layer frame.pair_inverse_order frame.repeated_parts frame.row_inverse'
                  ' frame.row_order frame.suffix_parts frame.w21 frame.weight').split(),
        provides=('state.birth_history state.row_inverse_prev').split(),
        nonterminal_only=True),
}

STAGE_ORDER = tuple(STAGE_FIELDS)
PIPELINE_CONFIG = dict.fromkeys(STAGE_ORDER, 'default')


def identity_stage(state, frame):
    """Keep the existing field values for an optional correction/update."""


def diagonal_transport(state, frame):
    previous = state.runtime.transport_mode
    state.runtime.transport_mode = 'diagonal'
    try:
        prepare_layer(state, frame)
    finally:
        state.runtime.transport_mode = previous


def zero_k3_legs(state, frame):
    frame.pre_transport = None
    frame.k3_diagonal = state.zeros_vector
    frame.k3_21 = None if frame.is_final else fnp.zeros((state.width, state.width), dtype=fnp.float32)
    frame.transport_alive = None
    frame.groups = []
    frame.suffix_parts = []
    frame.repeated_parts = []


def zero_k4_host_stars(state, frame):
    frame.radial_diagonal = frame.variance_pre
    frame.k4_diagonal = state.zeros_vector
    frame.radial_k4 = frame.k4_diagonal
    frame.k4_22 = None if frame.is_final else fnp.zeros((state.width, state.width), dtype=fnp.float32)


def disable_joint_geometry(state, frame):
    frame.has_joint = False
    for field in ('joint_h', 'joint_heads', 'joint_sigma', 'geometry_delta', 'geometry_heads'):
        setattr(frame, field, None)


def disable_v2e_r8(state, frame):
    frame.mean_delta = None
    state.corrector_carry = None


def gaussian_relu_moments(state, frame):
    """Gaussian marginal/pair readout in the same moment representation."""
    frame.k3_diagonal = state.zeros_vector
    frame.k4_diagonal = state.zeros_vector
    if not frame.is_final:
        frame.k3_21 = fnp.zeros((state.width, state.width), dtype=fnp.float32)
        frame.k4_22 = fnp.zeros((state.width, state.width), dtype=fnp.float32)
    relu_moments(state, frame)


def disable_source_birth(state, frame):
    state.carried_gain = frame.w11
    frame.w01, frame.w31 = frame.wick[W01], frame.wick[W31]
    frame.alive = None
    frame.transport_alive = None


def zero_k4_vertices(state, frame):
    state.source_k4_birth_ratios.append(state.k4_family_ratios)
    n = state.width if frame.alive is None else int(frame.alive.shape[0])
    zero = fnp.zeros(n, dtype=fnp.float32)
    if state.source_k4_star_vertices is not None:
        state.source_k4_star_vertices.append(zero)
    if state.source_k4_family_vertices is not None:
        state.source_k4_family_vertices.append((zero, zero, zero))


def diagonal_mean_cov(state, frame):
    state.mean = frame.p1 if frame.mean_delta is None else frame.p1 + frame.mean_delta
    state.covariance = fnp.diag(frame.k2_diagonal)


def disable_history(state, frame):
    state.runtime.feed_pending.clear()
    state.rows.append(state.mean)


def finish_history_without_feed(state, frame):
    previous = state.feed_active
    state.feed_active = False
    try:
        finish_history(state, frame)
        state.runtime.feed_pending.clear()
    finally:
        state.feed_active = previous

STAGE_IMPLEMENTATIONS = {
    'transport': {'default': prepare_layer, 'diagonal': diagonal_transport},
    'k3_legs': {'default': transport_cumulants3, 'off': zero_k3_legs},
    'k4_host_stars': {'default': propagate_cumulants4, 'off': zero_k4_host_stars},
    'joint_geometry': {'default': apply_joint_geometry, 'off': disable_joint_geometry},
    'h12_fe30': {'default': apply_birth_diagonal, 'off': identity_stage},
    'k3_calibration': {'default': calibrate_k3_level, 'off': identity_stage},
    'v2e_r8': {'default': apply_corrector, 'off': disable_v2e_r8},
    'relu_moments': {'default': relu_moments, 'gaussian': gaussian_relu_moments},
    'source_birth': {'default': prepare_source_birth, 'off': disable_source_birth},
    'k4_vertices': {'default': store_k4_vertices, 'off': zero_k4_vertices},
    'source_storage': {'default': store_source_birth, 'off': identity_stage},
    'post_corrections': {'default': correct_post_moments, 'off': identity_stage},
    'host_update': {'default': update_k4_host, 'off': identity_stage},
    'mean_cov': {'default': finish_moments, 'diagonal': diagonal_mean_cov},
    'history_feed': {'default': finish_history, 'off': disable_history, 'no_feed': finish_history_without_feed},
}

STAGE_ALIASES = {choices['default'].__name__: name
                 for name, choices in STAGE_IMPLEMENTATIONS.items()}

def stage_contracts():
    registry = {}
    overrides = {
        ('k3_legs', 'off'): (('state.width', 'state.zeros_vector', 'frame.is_final'),
                            ('frame.pre_transport', 'frame.k3_diagonal', 'frame.k3_21', 'frame.transport_alive', 'frame.groups', 'frame.suffix_parts', 'frame.repeated_parts')),
        ('source_birth', 'off'): (('frame.w11', 'frame.wick'),
                                 ('state.carried_gain', 'frame.w01', 'frame.w31', 'frame.alive', 'frame.transport_alive')),
        ('source_storage', 'off'): ((), ()),
        ('history_feed', 'off'): (('state.mean',), ('state.rows',)),
        ('k4_vertices', 'off'): (('state.width', 'frame.alive', 'state.source_k4_star_vertices', 'state.source_k4_family_vertices',
                                 'state.source_k4_birth_ratios', 'state.k4_family_ratios'),
                                ('state.source_k4_star_vertices', 'state.source_k4_family_vertices', 'state.source_k4_birth_ratios')),
    }
    for stage, implementations in STAGE_IMPLEMENTATIONS.items():
        registry[stage] = {}
        for name, function in implementations.items():
            fields = STAGE_FIELDS[stage]
            requires, provides = overrides.get((stage, name), (tuple(fields['requires']), tuple(fields['provides'])))
            if function is identity_stage:
                if stage == 'source_storage':
                    requires, provides = (), ()
                else:
                    # Optional corrections retain fields produced by earlier stages.
                    requires, provides = tuple(fields['provides']), tuple(fields['provides'])
            registry[stage][name] = dict(function=function, requires=requires, provides=provides,
                                        nonterminal_only=fields['nonterminal_only'])
    return registry


STAGE_REGISTRY = stage_contracts()


def validate_pipeline(config=None):
    choices = dict(PIPELINE_CONFIG)
    if config:
        for name, implementation in config.items():
            name = STAGE_ALIASES.get(name, name)
            if name not in choices:
                raise ValueError('Unknown pipeline stage: ' + name)
            if implementation not in STAGE_IMPLEMENTATIONS[name]:
                raise ValueError('Unknown implementation for ' + name + ': ' + str(implementation))
            choices[name] = implementation
    bank_stages = ('k3_legs', 'source_birth', 'source_storage', 'history_feed')
    if any(choices[name] == 'off' for name in ('k3_legs', 'source_birth', 'source_storage', 'history_feed')):
        if not all(choices[name] == 'off' for name in bank_stages):
            raise ValueError('Disabling the source bank requires k3_legs, source_birth, source_storage and history_feed all off')
    available = {'frame.layer', 'frame.weight'}
    available.update(field for name in STAGE_ORDER for field in STAGE_FIELDS[name]['requires'] if field.startswith('state.'))
    for name in STAGE_ORDER:
        fields = STAGE_REGISTRY[name][choices[name]]
        missing = set(fields['requires']) - available
        if missing:
            raise ValueError('Unsatisfied stage fields for ' + name + ': ' + ', '.join(sorted(missing)))
        available.update(fields['provides'])
    return choices


def compile_pipeline(runtime, config=None):
    runtime.pipeline_config = validate_pipeline(config)
    runtime.pipeline = tuple((name, STAGE_IMPLEMENTATIONS[name][runtime.pipeline_config[name]]) for name in STAGE_ORDER)
    runtime.transport_mode = 'default'
    if not hasattr(runtime, 'observer'):
        runtime.observer = None


def run_layer_pipeline(state, layer, weight):
    frame = LayerFrame(layer=layer, weight=weight)
    observer = state.runtime.observer
    for name, implementation in state.runtime.pipeline:
        if observer is not None:
            observer(name, 'before', state, frame)
        implementation(state, frame)
        if observer is not None:
            observer(name, 'after', state, frame)
        if name == 'relu_moments' and frame.is_final:
            return
