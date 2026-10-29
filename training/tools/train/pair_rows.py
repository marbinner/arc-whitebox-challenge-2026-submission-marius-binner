"""Compare declared paired eval rows using the canonical MSE summary."""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def rows(pattern):
    result={}
    for filename in sorted(glob.glob(pattern)):
        for row in json.loads(Path(filename).read_text())['rows']:
            index=row['idx']
            if type(index) is not int: raise ValueError('Paired panel identities must be integers')
            if index in result and result[index]!=row: raise ValueError('Conflicting duplicate paired row: '+str(index))
            if not np.isfinite(row['full']) or row['full']<0: raise ValueError('Invalid paired MSE')
            result[index]=row
    return result


def compare(a,b):
    if not a or not b: raise ValueError('Paired panel is empty')
    if set(a)!=set(b): raise ValueError('Paired panel identities differ')
    indices=sorted(a);verified=True
    for index in indices:
        for key in ('dataset_key','weights_sha256','target_sha256','seed','noise'):
            if a[index].get(key)!=b[index].get(key): raise ValueError('Paired row data identity differs: '+key)
        verified &= (type(a[index].get('seed')) is int and 0<=a[index]['seed']<2**64 and
                     all(isinstance(a[index].get(key),str) and len(a[index][key])==64 and
                         all(char in '0123456789abcdef' for char in a[index][key])
                         for key in ('weights_sha256','target_sha256')))
    av=np.array([a[i]['full'] for i in indices]);bv=np.array([b[i]['full'] for i in indices])
    noise=np.array([a[i].get('noise',0.0) for i in indices])
    if not np.isfinite(noise).all() or (noise<0).any(): raise ValueError('Invalid paired label noise')
    ratio=lambda x,y:float(x.sum()/y.sum()) if len(y) and y.sum()>0 else None
    if (bv>0).all():
        from tools.common import phase2_root
        sys.path.insert(0,str(phase2_root()/'experiments/bench'))
        import whest_bench
        summary=whest_bench.paired_summary(bv,av,noise=noise)
    else:
        summary=dict(rows=len(indices),wins=int((av<bv).sum()),paired_gain_z=None,
                     paired_gain_undefined='zero baseline row')
    midpoint=len(indices)//2
    return dict(indices=indices,ratio=ratio(av,bv),noise_corrected_ratio=ratio(av-noise,bv-noise),
                summary=summary,halves=[ratio(av[:midpoint],bv[:midpoint]),ratio(av[midpoint:],bv[midpoint:])],
                data_identity_verified=bool(verified),canonical_summary='whest_bench.paired_summary')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',required=True);parser.add_argument('--b',required=True)
    parser.add_argument('--label',default='a vs b')
    args=parser.parse_args(argv)
    report=dict(label=args.label,**compare(rows(args.a),rows(args.b)))
    print(json.dumps(report,allow_nan=False))
    return report


if __name__=='__main__':main()
