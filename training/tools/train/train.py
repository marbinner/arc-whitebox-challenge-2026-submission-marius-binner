"""Compatibility entry point; the only trainer implementation is run.py."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.train.run import main


def compatibility_arguments(argv):
    """CT's existing adapter supplies a bulk directory as --out."""
    args=list(argv)
    for index,value in enumerate(args):
        if value=='--out' and index+1<len(args):
            path=Path(args[index+1])
            if path.is_dir() or path.suffix!='.pt':
                args[index+1]=str(path/'train.pt')
        elif value.startswith('--out='):
            path=Path(value.split('=',1)[1])
            if path.is_dir() or path.suffix!='.pt':
                args[index]='--out='+str(path/'train.pt')
    return ['train',*args]


if __name__=='__main__': main(compatibility_arguments(sys.argv[1:]))
