"""Reusable acceptance checks; fresh stock namespaces are mandatory in G1."""
import math

import numpy as np
import torch


def relative(left,right):
    left=torch.as_tensor(left).double().cpu();right=torch.as_tensor(right).double().cpu()
    return float((left-right).norm()/right.norm().clamp_min(1e-30))


def fresh_namespace_gate(factory,panels,request,loader,tolerance=2e-6):
    injected=factory()
    rows=[]
    with torch.no_grad():
        for net in panels:
            weights=loader(net)[0]
            stock=factory().forward(weights)
            prediction=injected.forward(weights,request())
            repeated=factory().forward(weights)
            row=dict(net=net,relative_all=relative(prediction,stock),
                     relative_final=relative(prediction[-1],stock[-1]),
                     stock_repeat_relative=relative(repeated,stock))
            row['passed']=max(row['relative_all'],row['relative_final'],row['stock_repeat_relative'])<=tolerance
            rows.append(row)
    return dict(check='G1 fresh stock namespace per network',passed=all(row['passed'] for row in rows),
                tolerance=tolerance,rows=rows)


def compare_predictions(tensor,native,tolerance=1e-5):
    array=tensor.detach().cpu().numpy()
    return dict(relative_all=relative(array,native),relative_final=relative(array[-1],native[-1]),
                max_abs=float(np.max(np.abs(array.astype(np.float64)-native.astype(np.float64)))),
                finite=bool(np.isfinite(array).all() and np.isfinite(native).all()),
                passed=bool(np.isfinite(array).all() and np.isfinite(native).all() and
                            max(relative(array,native),relative(array[-1],native[-1]))<=tolerance))


def compare_gradient(left,right,*,rtol=0.01,atol=1e-8,cosine_min=0.999):
    left=left.detach().cpu().double().reshape(-1);right=torch.as_tensor(right).double().reshape(-1)
    if left.shape!=right.shape: return dict(passed=False,error='gradient shape mismatch')
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()): return dict(passed=False,error='nonfinite gradient')
    if float(right.norm())==0: return dict(passed=float(left.norm())==0,reference_zero=True)
    cosine=float(left@right/(left.norm()*right.norm()).clamp_min(1e-300))
    error=float((left-right).norm());reference_norm=float(right.norm())
    magnitude_passed=error<=atol+rtol*reference_norm
    return dict(cosine=cosine,relative_norm=error/reference_norm,absolute_error=error,
                norm=float(left.norm()),reference_norm=reference_norm,
                rtol=rtol,atol=atol,magnitude_passed=magnitude_passed,
                passed=cosine>=cosine_min and magnitude_passed)


def refresh_gate(factory,weights,changes):
    """A→B→A and fresh-B parity catch eval caches left by training/EMA changes."""
    rows=[]
    for parameter,(old,new) in changes.items():
        model=factory()
        with torch.no_grad():
            a=model.forward(weights,dict(params={parameter:old}))
            b=model.forward(weights,dict(params={parameter:new}))
            fresh=factory().forward(weights,dict(params={parameter:new}))
            restored=model.forward(weights,dict(params={parameter:old}))
        active=bool(torch.any(a!=b))
        row=dict(parameter=parameter,changed_prediction=active,
                 fresh_equal=bool(torch.equal(b,fresh)),restored_equal=bool(torch.equal(a,restored)))
        row['passed']=active and row['fresh_equal'] and row['restored_equal'];rows.append(row)
    return dict(check='ew34/heavy/joint evaluation refresh',rows=rows,passed=all(row['passed'] for row in rows))
