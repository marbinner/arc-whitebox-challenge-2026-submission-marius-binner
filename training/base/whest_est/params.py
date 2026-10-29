"""Registry-derived configuration; build substitutes native literal maps.

Inputs are registered defaults; outputs are independent PARAMS/PARAM_SPACE maps.
Kernel storage controls stay fixed. Refresh dispatch after runtime overrides.
"""
from __future__ import annotations
from .registry import parameter_defaults, parameter_space

PARAMS = parameter_defaults()
PARAM_SPACE = parameter_space()
KERNEL_CONSTANTS = {'block_output_max_elements': 3145728}
KERNEL_CONFIG = {**KERNEL_CONSTANTS, **PARAMS}

def refresh_kernel_config():
    KERNEL_CONFIG.clear()
    KERNEL_CONFIG.update(KERNEL_CONSTANTS)
    KERNEL_CONFIG.update(PARAMS)
