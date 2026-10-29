"""Fresh production state and fixed source-bank geometry.
Math: mean=0, covariance=I, radial host=-6/(width+2), chi mean factor.
Inputs: owned runtime and MLP. Outputs: empty source/history banks and LayerState.
Knobs: registered gains, coefficients and tier defaults."""
from __future__ import annotations
from .config import PARAMS
from .cumulants3 import fixed_projection
from .moments import chi_mean_ratio
from .selection import parse_tiers
from .state import LayerState, fnp

def initialize_state(runtime, mlp):
    _fe42_sel = None
    leg_tiers = parse_tiers(PARAMS['leg_tiers'])
    if len(leg_tiers) != 1:
        raise ValueError('static package supports one nonempty leg tier')
    runtime.feed_pending.clear()
    runtime.feed_active = int(PARAMS['feed_fold']) != 0
    runtime.history_join['ctx'] = None
    width = mlp.width
    zeros_vector = fnp.zeros(width, dtype=fnp.float32)
    ones_vector = fnp.ones(width, dtype=fnp.float32)
    wick_coefficients = runtime.coefficients['wick']
    power_mix = runtime.coefficients['power_mix']
    chi_mean = chi_mean_ratio(width)
    remainder_projection = fixed_projection(runtime, width, min(width, 2))
    transport_tiers = parse_tiers(PARAMS['transport_tiers'])
    row_alive_tiers = parse_tiers(PARAMS['row_alive_tiers'])
    col_alive_tiers = parse_tiers(PARAMS['col_alive_tiers'])
    row_inverse_prev = None
    leg_level_gain = float(PARAMS['leg_level_gain'])
    final_k3_scale = float(PARAMS['final_k3_scale'])
    final_k4_scale = float(PARAMS['final_k4_scale'])
    terminal_kink_rows = min(int(PARAMS['terminal_kink_rows']), width)
    if terminal_kink_rows <= 0:
        raise ValueError('terminal_kink_rows must be positive')
    pair_kink_rows = min(int(PARAMS['pair_kink_rows']), width)
    if pair_kink_rows <= 0:
        raise ValueError('pair_kink_rows must be positive')
    carrier_terms = [('mu_var', float(PARAMS['k3_carrier_mu_var']))] if PARAMS['k3_carrier_mu_var'] != 0.0 else []
    dress_rate = float(PARAMS['k3_vertex_dress']) / 6.0
    pair_ck3 = float(PARAMS['pair_ck3'])
    cov_k3sq = float(PARAMS['cov_k3sq']) / 6.0
    cov_k3cross = float(PARAMS['cov_k3cross']) / 6.0
    t2_k3_scale = float(PARAMS['t2_k3_scale'])
    _fe34_t2 = PARAMS['t2_k3_shape']
    t2_k3_coefficients = fnp.asarray((0.25 * t2_k3_scale * _fe34_t2[0], 0.5 * t2_k3_scale * _fe34_t2[1], 0.25 * t2_k3_scale * _fe34_t2[2]), dtype=fnp.float32)
    split_pair_c3_cov = float(PARAMS['split_pair_c3_cov'])
    cov_adt3 = float(PARAMS['cov_adt3'])
    cov_weight = float(PARAMS['k4_cov_weight'])
    identity_share = 1.0 - cov_weight
    cov_frobenius_share = cov_weight * cov_weight
    identity_frobenius_share = 1.0 - cov_frobenius_share
    identity_gain = 0.0
    cov_gain = cov_weight
    mean = zeros_vector
    covariance = fnp.eye(width, dtype=fnp.float32)
    carried_gain = None
    source_path_factors = []
    source_free_u = []
    source_free_diagonals = []
    source_remainder_left = []
    source_remainder_right = []
    source_singleton_diagonals = []
    source_alive = []
    _fe10_qoff = []
    source_transport_alive = []
    k4_star_scale = float(PARAMS.get('k4_star_scale', 0.0))
    k4_22_dress = float(PARAMS.get('k4_22_dress', 0.0))
    source_k4_star_vertices = [] if k4_star_scale != 0.0 else None
    k4_family_ratios = tuple((float(PARAMS.get(key, 0.0)) / k4_star_scale for key in ('k4_d22c_scale', 'k4_d31c_scale'))) + (1.0 / k4_star_scale,) if k4_star_scale != 0.0 else (0.0, 0.0, 0.0)
    source_k4_family_vertices = [] if source_k4_star_vertices is not None and any((ratio != 0.0 for ratio in k4_family_ratios)) else None
    leg_dense = []
    leg_q_dense = []
    leg_left = []
    leg_right = []
    leg_path_right = []
    k4_core = -6.0 / float(width + 2)
    _corr_carry = None
    runtime.features.clear()
    runtime.older_births.clear()
    runtime.birth_diagonal.clear()
    _h12_history = []
    _h12_self_c = []
    _h12_c2 = []
    rows = []
    state = LayerState(feed_active=runtime.feed_active, corrector_carry=_corr_carry, q_offsets=_fe10_qoff, covariance_selection=_fe42_sel, birth_c2=_h12_c2, birth_history=_h12_history, birth_self_c=_h12_self_c, carried_gain=carried_gain, carrier_terms=carrier_terms, col_alive_tiers=col_alive_tiers, cov_adt3=cov_adt3, cov_frobenius_share=cov_frobenius_share, cov_gain=cov_gain, cov_k3cross=cov_k3cross, cov_k3sq=cov_k3sq, cov_weight=cov_weight, covariance=covariance, dress_rate=dress_rate, final_k3_scale=final_k3_scale, final_k4_scale=final_k4_scale, identity_frobenius_share=identity_frobenius_share, identity_gain=identity_gain, identity_share=identity_share, k4_22_dress=k4_22_dress, k4_core=k4_core, k4_family_ratios=k4_family_ratios, k4_star_scale=k4_star_scale, leg_dense=leg_dense, leg_left=leg_left, leg_level_gain=leg_level_gain, leg_path_right=leg_path_right, leg_q_dense=leg_q_dense, leg_right=leg_right, leg_tiers=leg_tiers, mean=mean, ones_vector=ones_vector, pair_ck3=pair_ck3, pair_kink_rows=pair_kink_rows, power_mix=power_mix, remainder_projection=remainder_projection, row_alive_tiers=row_alive_tiers, row_inverse_prev=row_inverse_prev, rows=rows, source_alive=source_alive, source_free_diagonals=source_free_diagonals, source_free_u=source_free_u, source_k4_family_vertices=source_k4_family_vertices, source_k4_star_vertices=source_k4_star_vertices, source_path_factors=source_path_factors, source_remainder_left=source_remainder_left, source_remainder_right=source_remainder_right, source_singleton_diagonals=source_singleton_diagonals, source_transport_alive=source_transport_alive, split_pair_c3_cov=split_pair_c3_cov, t2_k3_coefficients=t2_k3_coefficients, t2_k3_scale=t2_k3_scale, terminal_kink_rows=terminal_kink_rows, transport_tiers=transport_tiers, wick_coefficients=wick_coefficients, width=width, zeros_vector=zeros_vector, mlp=mlp, runtime=runtime)
    state.source_padding = {}
    state.source_k4_birth_ratios = []
    state.chi_mean = chi_mean
    return state
