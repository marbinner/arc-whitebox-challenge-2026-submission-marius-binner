"""Public configuration aliases for registry-derived parameters.
Inputs: registered defaults and runtime overrides. Outputs: parameter maps and
kernel dispatch settings. No mathematical work; knobs are defined in registry.py."""
from .params import PARAMS, PARAM_SPACE, KERNEL_CONSTANTS, KERNEL_CONFIG, refresh_kernel_config
