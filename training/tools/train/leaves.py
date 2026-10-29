"""Compatibility imports. All model updates now use the public runtime API."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.train.folding import DEPTH,WIDTH,fold_corrector,layer_weights,offload_saved
from tools.train.runtime import DEFAULT_BASE,RuntimeTwin
from tools.train.state import TrainState
GradTwin=RuntimeTwin
