#!/usr/bin/env python
"""One command for clean-base tune, train, export and eval."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='tools.train'

import torch
from . import data
from .engine import Engine,validate_recipe
from .checkpoint import CheckpointWeights, warm_start_checkpoint
from .export import export_state, import_request, write_json
from .runtime import BUDGET, DEFAULT_BASE, NativeRuntime, RuntimeTwin,merge_configuration
from .selection import add_joint
from .state import TrainState
from ..guards import Paths
normalize_paths = Paths.normalize_paths


# val selects the best checkpoint; report_val is evaluated at the same steps for an honest read but never selects.
# Both accept 'd8b:'-prefixed ranges (e.g. --val d8b:12536:13048 --report-val 832:864).
DEFAULTS=dict(train='0:800',d8b='0:3072',train_rep=1,val='832:864',report_val='',seed=33,
              lr_corr=1e-5,lr_v2e=3e-4,lr_knobs=1e-3,lr_beta=5e-4,
              lr_k3band=1e-4,lr_fe31=3e-4,lr_joint=0.0,lr_scale=1.0,grow_lr=1e-4,knob_floor=0.3,
              sched_offset=0,ema=0.995,ema_warmup=1,offload_mb=0.0,offload_arena_gb=0.0,ml_weight=0.0,accum=1,
              loss_norm='net',s0=1.2e-8)


def csv(values):
    return [part.strip() for value in values or () for part in value.split(',') if part.strip()]


def common(parser):
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE,help='Clean scientific source package (base/)')
    parser.add_argument('--device',default=None,help='cpu or cuda; default CUDA when available')
    parser.add_argument('--norms',default=os.environ.get('WHEST_TRAIN_NORMS'),help='f3 corrector_norms.npz, needed for legacy checkpoints')
    parser.add_argument('--runtime-config',type=Path,help='Optional public API configuration (pipeline, corrector, per_layer, params)')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('tune','train'):
        q=sub.add_parser(command,help='Tune continuous knobs' if command=='tune' else 'Refit registered corrector/assets/knobs')
        common(q)
        init=q.add_mutually_exclusive_group();init.add_argument('--init',type=Path);init.add_argument('--resume',type=Path)
        q.add_argument('--init-weights',choices=('live','ema'),default='live',
                       help='Warm-start weights; _best.pt selection uses EMA, while legacy initialization uses live')
        q.add_argument('--steps',type=int,default=3000,help='Additional optimizer steps; resume retains schedule and sampler')
        q.add_argument('--eval-every',type=int,default=500)
        q.add_argument('--out',type=Path,required=True,help='Live continuation checkpoint; _best.pt holds best validation state')
        for key,value in DEFAULTS.items():
            q.add_argument('--'+key.replace('_','-'),type=type(value),default=None,
                           choices=('net','global') if key=='loss_norm' else None)
        q.add_argument('--d8c',default=None,help='Additional d8c training ranges, e.g. 0:4096; legacy defaults stay unchanged')
        q.add_argument('--packed-checkpoint',action='store_true',default=None,help='Recompute packed products during backward; requires source/memory qualification')
        q.add_argument('--sched-total',type=int,default=None)
        q.add_argument('--base-cache',type=Path)
        q.add_argument('--test',default='',help='Legacy recipe metadata; test nets are opened only by explicit eval')
        q.add_argument('--monitor-train',type=int,default=0,help='Compare live loss before/after on this many deterministic training-panel rows')
        q.add_argument('--in-sample',action='store_true',help='Explicitly label training on reserved/evaluation panels as in-sample')
        q.add_argument('--grow',default='',help='Function-preserving ff=256,nb=3; active runtime schema must support the new shapes')
        q.add_argument('--grow-init',choices=('zero','split'),default='zero',
                       help='New FF units: zero output weights (default) or split existing units (Net2Wider, active at once)')
        q.add_argument('--tune-all',action='store_true',help='Include every registry continuous trainable PARAMS key')
        q.add_argument('--name','--names',action='append',default=[],help='Registry names/globs, e.g. edgeworth_34 or corrector.Wf0')
        q.add_argument('--group','--groups',action='append',default=[],help='Registry groups, e.g. corrector,v2e,fe31,params')
        q.add_argument('--owner','--owners',action='append',default=[],help='Registry scientific owner, e.g. moments,history')
        q.add_argument('--per-layer',action='append',default=[],help='Expand selected scalar parameters to length-16 schedules')
        q.add_argument('--lr',action='append',default=[],help='Override a named learning rate: PARAMS-key=rate or tensor-name=rate')
        q.add_argument('--train-zero-knobs',action='store_true',
                       help='Also train selected knobs that start at exactly 0 (default: they stay 0; a zero gain disables its native branch)')
    q=sub.add_parser('export',help='Export a checkpoint to C-order assets and PARAMS_DIFF.json')
    common(q);q.add_argument('--ckpt',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--weights',choices=('ema','live'),default='ema')
    q=sub.add_parser('eval',help='Evaluate checkpoints, stock or native exports on an explicit panel')
    common(q);q.add_argument('--ckpt',type=Path);q.add_argument('--stock',action='store_true')
    q.add_argument('--native',action='store_true',help='Metered native backend; records FLOPs and counted operations')
    q.add_argument('--export',type=Path,help='Import this export through runtime API v1')
    q.add_argument('--legacy-export',action='store_true',help='Explicitly accept exports without the current completion manifest')
    q.add_argument('--weights',choices=('ema','live'),default='ema')
    q.add_argument('--nets',default='');q.add_argument('--d8b-nets',default='');q.add_argument('--d8c-nets',default='');q.add_argument('--out',type=Path,required=True)
    return p


def configuration(args):
    if not hasattr(args, '_runtime_configuration'):
        args._runtime_configuration = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    return args._runtime_configuration


def device(args):
    return args.device or ('cuda' if torch.cuda.is_available() else 'cpu')


def load_state(args):
    checkpoint=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    config=merge_configuration(checkpoint.get('config',{}).get('runtime_config') or {},configuration(args))
    twin=RuntimeTwin(args.base,device(args),config=config,
                     **({'packed_checkpoint':True} if checkpoint.get('config',{}).get('packed_checkpoint',False) else {}))
    state=TrainState(twin,checkpoint,args.norms)
    with torch.no_grad(): twin.model.update_runtime(**state.request(evaluation=True))
    state.refresh_schema(twin)
    engine=None
    if checkpoint.get('schema')=='whest-train-v1':
        engine=CheckpointWeights(state,checkpoint)
    return twin,state,engine


def stock_cache_rows(rows, panel):
    """Validate all cached loss denominators before updating the live cache."""
    if not isinstance(rows, dict):
        raise ValueError('Stock MSE cache rows must be an object')
    allowed, validated = set(panel), {}
    for key, value in rows.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError('Invalid Stock MSE cache network identity') from None
        if (type(key) not in (str, int) or str(index) != str(key) or index not in allowed
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid Stock MSE cache row: ' + repr(key))
        validated[index] = value
    return validated


def stock_data_identity(engine):
    """Labels bind corpus files; these bytes also bind canonical MLP regeneration."""
    if getattr(engine, 'loader', data.net) is not data.net:
        raise ValueError('Shared v3 stock cache requires the canonical validated loader')
    files = {'canonical_data.py': Path(data.canonical.__file__),
             'train_data.py': Path(data.__file__)}
    return dict(schema='whest-stock-data-v1',
                files_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in sorted(files.items())})


def load_stock_cache(cached, twin, engine, panel):
    """A stock prediction depends on the model and labels, never the optimizer recipe."""
    if not isinstance(cached, dict):
        raise ValueError('Stock MSE cache must be an object')
    if cached.get('schema') not in ('whest-stock-mse-v2', 'whest-stock-mse-v3'):
        raise ValueError('Stock MSE cache lacks current execution identities; rebuild it')
    if cached.get('source_sha256') != twin.source_sha256:
        raise ValueError('Stock MSE cache has a different base source')
    if cached.get('model_identity') != getattr(twin, 'identity', None):
        raise ValueError('Stock MSE cache has different assets or runtime configuration; rebuild it')
    if cached['schema'] == 'whest-stock-mse-v2':
        # Legacy files remain strict: do not reinterpret their label binding.
        if cached.get('recipe_identity') != engine.recipe_identity or cached.get('dataset_identity') != engine.dataset_identity:
            raise ValueError('Stock MSE cache has a different trainer recipe or label corpus; rebuild it')
        return stock_cache_rows(cached.get('rows'), panel)
    expected_data = stock_data_identity(engine)
    cached_data = cached.get('data_identity')
    if cached_data is None:
        # The first v3 writer already records these module bytes in its full
        # recipe receipt. Bind only those numeric inputs, not rates/schedule.
        files = cached.get('recipe_identity', {}).get('files', {})
        cached_data = dict(schema='whest-stock-data-v1', files_sha256={
            'canonical_data.py': files.get('../data.py'),
            'train_data.py': files.get('data.py')})
    if cached_data != expected_data:
        raise ValueError('Stock MSE cache has different weight-regeneration code; rebuild it')
    corpus = cached.get('dataset_identity')
    if not isinstance(corpus, dict) or corpus.get('schema') != 'whest-label-corpus-v2' or not corpus.get('sources'):
        raise ValueError('Stock MSE cache lacks a bound label corpus')
    indices = data.validate_evaluation_panel(corpus.get('indices', []))
    data.verify_corpus_identity(corpus)
    current = {source['path']: source for source in engine.dataset_identity.get('sources', ())}
    for source in corpus['sources']:
        if source['path'] in current and source != current[source['path']]:
            raise ValueError('Stock MSE cache label version differs from the current corpus')
    validated = stock_cache_rows(cached.get('rows'), indices)
    allowed = set(panel)
    return {index: value for index, value in validated.items() if index in allowed}


def train(args):
    if args.steps<0 or args.eval_every<=0 or args.monitor_train<0:
        raise ValueError('Steps/monitor rows must be nonnegative and eval-every positive')
    names,groups,owners=csv(args.name),csv(args.group),csv(args.owner)
    if args.command=='tune' and not (names or groups or owners): groups=['params']
    if args.tune_all and 'params' not in groups: groups.append('params')
    ck=torch.load(args.resume or args.init,map_location='cpu',weights_only=False) if args.resume or args.init else None
    if args.init_weights != 'live' and not args.init:
        raise ValueError('--init-weights applies only to --init; exact resume retains live optimizer state')
    if args.init:
        ck=warm_start_checkpoint(ck,args.init_weights)
    config=dict(DEFAULTS)
    if args.resume:
        if args.grow: raise ValueError('Exact resume cannot change corrector structure; use --init')
        if ck.get('schema')!='whest-train-v1': raise ValueError('Legacy files are warm starts; use --init')
        config=dict(ck['config'])
        if args.packed_checkpoint is not None and args.packed_checkpoint!=config.get('packed_checkpoint',False):
            raise ValueError('Exact resume cannot change --packed-checkpoint; use --init')
        if args.d8c is not None and args.d8c!=config.get('d8c',''):
            raise ValueError('Exact resume cannot change --d8c; use --init')
        if args.in_sample and not config.get('in_sample',False):
            raise ValueError('Exact resume cannot change in-sample labeling; use --init')
        if args.runtime_config and configuration(args)!=config.get('runtime_config'):
            raise ValueError('Exact resume cannot change runtime configuration; use --init')
        for key in DEFAULTS:
            supplied=getattr(args,key)
            if supplied is not None and supplied!=config.get(key):
                raise ValueError('Exact resume cannot change --'+key.replace('_','-')+'; use --init')
        if args.train_zero_knobs and not config.get('train_zero_knobs'):
            raise ValueError('Exact resume cannot change --train-zero-knobs; use --init')
        if args.sched_total is not None and args.sched_total!=config['sched_total']:
            raise ValueError('Exact resume cannot change the schedule; use --init')
        if names or groups or owners or args.lr or args.per_layer:
            # Implicit tune selection is harmless; explicit changes are warm starts.
            if not (args.command=='tune' and groups==['params'] and not names and not owners and not args.lr and not args.per_layer):
                raise ValueError('Exact resume retains registry selection and learning rates; use --init')
    else:
        config.update({key:getattr(args,key) for key in DEFAULTS if getattr(args,key) is not None})
        if config['train_rep']<1: raise ValueError('train-rep must be positive')
        config['population']=data.ranges(config['train'])*config['train_rep']+data.ranges(config['d8b'],d8b=True)
        if args.d8c:
            config['d8c']=args.d8c
            config['population']+=data.ranges(args.d8c,d8c=True)
        config['sched_total']=max(args.steps,1) if args.sched_total is None else args.sched_total
        config['lr_overrides']={}
        for entry in args.lr:
            key,value=entry.rsplit('=',1);config['lr_overrides'][key]=float(value)
        config['runtime_config']=merge_configuration(
            (ck.get('config',{}).get('runtime_config') if ck else None) or {},configuration(args))
        config['in_sample']=args.in_sample
        config['train_zero_knobs']=args.train_zero_knobs
        config['packed_checkpoint']=bool(args.packed_checkpoint)
    if not args.resume:
        ancestry,known=data.training_ancestry(ck)
        source_provenance=data.package_provenance(args.base)
        config['training_ancestry']=sorted(set(config['population']+ancestry+source_provenance['seen_indices']))
        config['holdout_provenance_known']=known and (source_provenance['known'] if ck is None or source_provenance['seen_indices'] else True)
        config['ancestry_corpus_sha256']=sorted(set((ck or {}).get('training_provenance',{}).get('corpus_sha256',[])+source_provenance['corpus_sha256']))
    validate_recipe(config)
    if not 0<=config['ema']<1: raise ValueError('ema must be in [0,1)')
    if config.get('lr_joint',0)<0: raise ValueError('lr-joint must be nonnegative')
    if not args.resume and config.get('lr_joint',0)>0:
        # OB F0b: --lr-joint > 0 adds the joint/geometry heads (registry group 'joint') to the selection, at their own
        # rate; the default 0 leaves every older recipe unchanged (an explicit --group joint keeps the legacy lr_corr).
        groups=add_joint(names,groups,owners)
    if config['s0']<=0 or config['offload_mb']<0 or config['ml_weight']<0:
        raise ValueError('s0 must be positive; offload/auxiliary weights must be nonnegative')
    val,report_val=data.panel(config['val']),data.panel(config.get('report_val',''))
    if set(val)&set(report_val): raise ValueError('The report panel must be disjoint from the selection panel')
    data.validate_training_panel(config.get('training_ancestry',config['population']),val+report_val,
                                 in_sample=config.get('in_sample',False))
    if args.init: config['init_weights']=args.init_weights
    torch.manual_seed(config['seed'])
    twin=RuntimeTwin(args.base,device(args),config=config.get('runtime_config'),
                     **({'packed_checkpoint':True} if config.get('packed_checkpoint',False) else {}))
    state=TrainState(twin,ck,args.norms,names=() if args.resume else names,
                     groups=() if args.resume else groups,owners=() if args.resume else owners,
                     per_layer=csv(args.per_layer),warm_start=bool(args.init),
                     freeze_zero_knobs=not args.resume and not config.get('train_zero_knobs',False))
    if args.grow:
        grow={key:int(value) for key,value in (entry.split('=') for entry in args.grow.split(','))}
        if set(grow)-{'ff','nb'}: raise ValueError('grow accepts only ff and nb')
        state.grow(**grow,init=args.grow_init)
    # Validate growth and leaf mapping via the API before allocating a graph.
    with torch.no_grad():
        try: twin.model.update_runtime(**state.request(evaluation=True))
        except ValueError as exc:
            if args.grow: raise ValueError('--grow needs an active runtime schema supporting these shapes: '+str(exc)) from exc
            raise
    state.refresh_schema(twin,include_new_corrector=bool(args.grow or args.init))
    engine=Engine(twin,state,config)
    if args.resume: engine.restore(ck)
    if args.base_cache and args.base_cache.exists():
        cached=json.loads(args.base_cache.read_text())
        engine.base_cache.update(load_stock_cache(cached, twin, engine,
                                                  config['population'] + val + report_val))
    print(json.dumps(dict(runtime_api=1,selected=state.selected,optimizer_leaves=list(state.leaves),
                          n_train=len(config['population']),source_sha256=twin.source_sha256,
                          in_sample=config.get('in_sample',False))),flush=True)
    report=dict(schema='whest-train-run-v1',config=config,selected=state.selected,evals=[],
                in_sample=config.get('in_sample',False),
                holdout_status=('in-sample' if config.get('in_sample',False) else
                                'held-out' if config.get('holdout_provenance_known',True) else 'unknown'))
    monitor=list(dict.fromkeys(engine.sampler.order))[:args.monitor_train]
    def monitor_loss():
        rows=[]
        with torch.no_grad():
            for idx in monitor:
                weights,target,_=data.net(idx)
                value=float(engine.objective(idx,weights,target,evaluation=True))
                rows.append(dict(idx=idx,loss=value))
        return dict(n=len(rows),mean_loss=sum(row['loss'] for row in rows)/len(rows),rows=rows)
    if monitor:
        report['training_loss_before']=monitor_loss()
        print(json.dumps(dict(training_loss_before=report['training_loss_before']['mean_loss'],n=len(monitor))),flush=True)
    def evaluate():
        ev=engine.evaluate(val)
        if report_val: ev['report']=engine.evaluate(report_val)
        ev['step']=engine.completed;report['evals'].append(ev)
        metric=ev.get('ratio_noise_corrected')
        if metric is not None and (engine.best is None or metric<engine.best):
            engine.best=metric
            best_path=args.out.with_name(args.out.stem+'_best.pt').absolute()
            engine.best_artifact=None
            engine.save(best_path)
            engine.best_artifact=dict(path=str(best_path),sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
        report['best_artifact']=engine.best_artifact
        engine.save(args.out)
        write_json(args.out.with_suffix('.json'),dict(report,history=engine.history,completed=engine.completed))
        if args.base_cache:
            write_json(args.base_cache,dict(schema='whest-stock-mse-v3',source_sha256=twin.source_sha256,
                       model_identity=getattr(twin,'identity',None),recipe_identity=engine.recipe_identity,
                       data_identity=stock_data_identity(engine),
                       dataset_identity=engine.dataset_identity,rows=engine.base_cache))
        print(json.dumps(dict(step=engine.completed,val={k:v for k,v in ev.items() if k not in ('rows','report')},
                              **({'report':{k:v for k,v in ev['report'].items() if k!='rows'}} if report_val else {}))),flush=True)
    if not args.resume: evaluate()
    for index in range(args.steps):
        row=engine.step()
        if index%25==0: print(json.dumps(row),flush=True)
        if (index+1)%args.eval_every==0: evaluate()
    if args.steps==0 or args.steps%args.eval_every: evaluate()
    if monitor:
        report['training_loss_after']=monitor_loss()
        report['training_loss_lowered']=report['training_loss_after']['mean_loss']<report['training_loss_before']['mean_loss']
        write_json(args.out.with_suffix('.json'),dict(report,history=engine.history,completed=engine.completed))
        print(json.dumps(dict(training_loss_after=report['training_loss_after']['mean_loss'],lowered=report['training_loss_lowered'])),flush=True)
    print(json.dumps(dict(checkpoint=str(args.out),completed=engine.completed,status='complete')),flush=True)
    return report


def export(args):
    from contextlib import nullcontext
    twin,state,engine=load_state(args)
    with engine.use_weights(args.weights) if engine else nullcontext():
        receipt=export_state(twin,state,args.out,checkpoint=args.ckpt,config=configuration(args))
    print(json.dumps(dict(out=str(args.out),params_changed=receipt['params_changed'],npz_all_c_contiguous=True)))
    return receipt


def add_native_score(rows, budget=BUDGET):
    """Meter each row before averaging, as the competition objective requires."""
    import numpy as np
    for row in rows:
        row['compute_multiplier']=max(0.1,row['flops']/budget)
        row['adjusted_final_layer_score']=row['full']*row['compute_multiplier']
    return dict(adjusted_final_layer_score=float(np.mean([row['adjusted_final_layer_score'] for row in rows])),
                mean_effective_compute=float(np.mean([row['flops'] for row in rows])),
                worst_mlp_adjusted_final_layer_score=max(row['adjusted_final_layer_score'] for row in rows),
                flop_budget=budget)


def evaluate(args):
    from contextlib import nullcontext
    import numpy as np
    panel=data.validate_evaluation_panel(data.ranges(args.nets)+data.ranges(args.d8b_nets,d8b=True)+data.ranges(args.d8c_nets,d8c=True))
    if not panel: raise ValueError('Specify an evaluation panel with --nets or --d8b-nets')
    if not (args.stock or args.ckpt or args.export): raise ValueError('Specify --stock, --ckpt or --export')
    if args.stock and (args.ckpt or args.export) or args.ckpt and args.export: raise ValueError('Select one evaluation source')
    state,engine=None,None
    if args.ckpt: twin,state,engine=load_state(args)
    else: twin=RuntimeTwin(args.base,device(args),config=configuration(args))
    native=NativeRuntime(args.base) if args.native else None
    rows=[]
    with engine.use_weights(args.weights) if engine else nullcontext(), torch.no_grad():
        request=merge_configuration(twin.config,state.request(evaluation=True)) if state else configuration(args) or {}
        if state: request=merge_configuration(request,configuration(args))
        if args.export:
            request=merge_configuration(import_request(args.export,twin.normalize_parameter,legacy=args.legacy_export),configuration(args))
            if not args.native:
                request['assets']={group:{key:torch.as_tensor(value,device=twin.device) for key,value in members.items()}
                                   for group,members in request['assets'].items()}
        if args.native and state:
            request['assets']={group:{key:np.array(value.detach().cpu().numpy(),order='C',copy=True) for key,value in members.items()}
                               for group,members in request['assets'].items()}
        for idx in panel:
            weights,target,noise,metadata=data.net_record(idx)
            if native:
                prediction,cost=native.forward(weights,request,seed=metadata['seed'])
                if prediction.shape!=(len(weights),len(target)) or not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite or invalid evaluation prediction on net '+str(idx))
                mse=float(np.mean((prediction[-1].astype(np.float64)-target.numpy())**2))
            else:
                prediction=twin.forward(weights,request,seed=metadata['seed']);cost={}
                if tuple(prediction.shape)!=(len(weights),len(target)) or not bool(torch.isfinite(prediction).all()):
                    raise ValueError('Nonfinite or invalid evaluation prediction on net '+str(idx))
                mse=float(((prediction[-1].double().cpu()-target)**2).mean())
            if not np.isfinite(mse) or mse<0:raise ValueError('Nonfinite evaluation prediction/MSE on net '+str(idx))
            rows.append(dict(metadata,idx=idx,full=mse,noise=noise,**cost))
            print(json.dumps(rows[-1]),flush=True)
    report=dict(schema='whest-train-eval-v1',backend='functional-native' if native else 'torch',
                base=str(args.base),checkpoint=str(args.ckpt) if args.ckpt else None,rows=rows,
                mse=float(np.mean([row['full'] for row in rows])),grader_certified=False)
    if native: report.update(add_native_score(rows))
    write_json(args.out,report)
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    normalize_paths(args)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    try:
        return {'train':train,'tune':train,'export':export,'eval':evaluate}[args.command](args)
    except (ValueError,FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__=='__main__': main()
