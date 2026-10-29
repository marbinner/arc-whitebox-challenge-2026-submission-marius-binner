"""Exact saved-tensor offload check through runtime API v1; no timing."""
import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='tools.train'

import torch
from . import data
from .export import write_json
from .runtime import DEFAULT_BASE,RuntimeTwin
from .state import TrainState
from .run import load_state
from ..guards import Paths
normalize_paths = Paths.normalize_paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--ckpt',type=Path,required=True)
    parser.add_argument('--norms',type=Path)
    parser.add_argument('--runtime-config',type=Path)
    parser.add_argument('--nets',default='800,809')
    parser.add_argument('--offload-mb',type=float,default=8.0)
    parser.add_argument('--offload-only',action='store_true',help='Memory diagnostic without identity comparison')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    normalize_paths(args)
    panel=data.validate_evaluation_panel(data.ranges(args.nets))
    if not panel: raise ValueError('Offload identity requires at least one network; empty panel')
    if not torch.cuda.is_available(): raise SystemExit('A CUDA device is required')
    if not math.isfinite(args.offload_mb) or args.offload_mb<=0: raise ValueError('offload-mb must be finite and positive')
    torch.set_num_threads(1)
    args.device='cuda'
    twin,state,_=load_state(args)
    def backward(weights,target,threshold):
        for value in state.leaves.values(): value.grad=None
        torch.cuda.reset_peak_memory_stats()
        prediction=twin.forward(weights,state.request(),offload_mb=threshold)
        loss=((prediction[-1].double()-target.to(twin.device))**2).mean()
        loss.backward()
        gradients={name:None if value.grad is None else value.grad.detach().cpu().clone()
                   for name,value in state.leaves.items()}
        return float(loss.detach()),gradients,int(torch.cuda.max_memory_allocated())
    rows=[]
    for idx in panel:
        weights,target,_=data.net(idx)
        plain=None if args.offload_only else backward(weights,target,0)
        offloaded=backward(weights,target,args.offload_mb)
        row=dict(net=idx,loss_offload=offloaded[0],peak_bytes_offload=offloaded[2],
                 n_tensors=len(offloaded[1]))
        if plain:
            different=[name for name,value in plain[1].items()
                       if (value is None)!=(offloaded[1][name] is None) or
                       value is not None and not torch.equal(value,offloaded[1][name])]
            row.update(loss_plain=plain[0],peak_bytes_plain=plain[2],
                       loss_identical=plain[0]==offloaded[0],different_gradients=different,
                       passed=plain[0]==offloaded[0] and not different)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(check='API saved-tensor offload identity',offload_mb=args.offload_mb,
                passed=None if args.offload_only else all(row['passed'] for row in rows),rows=rows,
                timing_measured=False)
    write_json(args.out,report)
    return report


def cli(argv=None):
    report=main(argv)
    return 1 if report['passed'] is False else 0


if __name__=='__main__': raise SystemExit(cli())
