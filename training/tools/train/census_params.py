"""Read the active public registry and optional optimizer-leaf census."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.train.runtime import DEFAULT_BASE,RuntimeTwin
from tools.train.selection import registry_entries,select
from tools.train.state import TrainState
from tools.train.run import load_state,configuration
from tools.guards import Paths
normalize_paths = Paths.normalize_paths


def census(twin,state=None):
    entries=registry_entries(twin.schema)
    selected=set(state.selected if state else select(twin.schema))
    return dict(runtime_api=twin.schema['version'],
                entries={key:dict(spec,selected=key in selected) for key,spec in entries.items()},
                optimizer_leaves={} if state is None else {
                    name:dict(shape=list(value.shape),dtype=str(value.dtype),numel=value.numel(),
                              registry_names=state.leaf_names[name]) for name,value in state.leaves.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--init',type=Path);parser.add_argument('--norms');parser.add_argument('--out',type=Path)
    parser.add_argument('--runtime-config',type=Path)
    args=parser.parse_args(argv);normalize_paths(args);args.device='cpu'
    if args.init:
        args.ckpt=args.init
        twin,state,_=load_state(args)
    else:
        twin=RuntimeTwin(args.base,'cpu',config=configuration(args));state=None
    payload=census(twin,state)
    if args.out:
        from tools.train.export import write_json
        write_json(args.out,payload)
    else: print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
