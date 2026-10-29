"""Compatibility entry point for run.py eval."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.train.run import main
if __name__=='__main__': main(['eval',*sys.argv[1:]])
