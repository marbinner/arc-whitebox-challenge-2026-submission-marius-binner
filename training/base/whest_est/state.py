"""Owned runtime caches, persistent prediction state and one-layer frames.
Inputs: parameters and fitted assets. Outputs: explicit state passed to stages.
Constants encode Gaussian derivatives and fixed geometry; knobs come from registry."""
from __future__ import annotations
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench import SetupContext
from whestbench.domain import MLP
import math
from .config import PARAMS
INV_SQRT_2PI = 0.3989422804014327
VAR_FLOOR = 1e-10
KERNEL_WIDTH = [1024]
ONE_THIRD = 0.3333333333333333
LOG_PI = 1.1447298858494002
(W01, W02, W03, W04, W11, W21, W31, W41, W51, W61) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
W71 = 10
WICK_COEFFICIENTS = ((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 1.0, 0.0, 0.0), (0.0, 5.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 1.0, 0.0, 0.0), (0.0, 3.0, 0.0, -1.0, 0.0), (3.0, 0.0, -6.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0, 0.0), (0.0, 3.0, 0.0, 1.0, 0.0), (3.0, 0.0, 6.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0))
SIXTH = 0.16666666666666666
POWER_MIX = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SIXTH, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08333333333333333, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
LEG_MEAN_FLOOR = 1e-30
CORRECTOR_KEYS = ('ABC', 'Gz', 'Wc0', 'Wc1', 'Wf0', 'Wf1', 'Wg0', 'Wg1', 'Wom', 'Wx', 'Wxf', 'Wxm', 'Wxr', 'Wxrm', 'bf0', 'bf1', 'bg0', 'bg1', 'bin', 'bom', 'bz')
HEAD_CONSTANT = 16.0
from types import SimpleNamespace

class Runtime(SimpleNamespace):
    """Owned setup models and per-prediction caches; explicitly passed to helpers."""

    def __init__(self):
        super().__init__(remainder_products=[], history_join={'ctx': None}, active_rows=[None], corrector_layers=None, v2e_model=None, v2e_output=None, feed_active=False, feed_pending=[], learned={}, readout={}, coefficients={}, older_births=[], birth_diagonal={}, joint_weights={}, projection=None, features={}, feature_layers=None, join_reused=[False], fused_join={})

class LayerState(SimpleNamespace):
    """Persistent mean/covariance, source banks, history, geometry and runtime.

    Arrays retain their native dtype and row domains. Components update only
    this shared state or the layer frame; estimator instances own their models.
    """
    birth_history: object
    covariance: object
    leg_dense: object
    leg_left: object
    leg_right: object
    mean: object
    rows: object
    source_path_factors: object
    runtime: object

class LayerFrame(SimpleNamespace):
    """One layer's preactivations, selections, cumulants, moments and source factors.

    Its lifetime is exactly one layer. Only values consumed by another stage
    are fields; temporary contraction operands stay local to their owner.
    """
    layer: object
    weight: object
    covariance_pre: object
    is_final: object
    k3_21: object
    k3_diagonal: object
    k4_22: object
    k4_diagonal: object
    mean_pre: object
    variance_pre: object
    wick: object
