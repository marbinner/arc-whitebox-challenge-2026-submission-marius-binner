"""Bounded original-coordinate warm cosine; EMA/SWA do not alter live training."""
import argparse
import copy
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import random
import shutil
import signal
import sys
import torch
import numpy as np
import rich.style

sys.path.insert(0,os.environ['WHEST_BASE'])
from tools.train import data
from tools.train.checkpoint import warm_start_checkpoint
from tools.train.engine import Engine
from tools.train.export import export_state,write_json
from tools.train.runtime import RuntimeTwin,cpu_tree
from tools.train.state import TrainState

DEADLINE=datetime(2099,1,1,0,0,0,tzinfo=timezone.utc)
RATES={
    'half-dc':dict(lr_corr=1e-4,lr_v2e=1.5e-4,lr_fe31=1.5e-4,lr_joint=5e-6,
                   lr_knobs=5e-4,lr_beta=2.5e-4,lr_k3band=5e-5),
    'ta-v3':dict(lr_corr=3e-5,lr_v2e=1e-4,lr_fe31=1e-4,lr_joint=1e-5,
                 lr_knobs=1e-3,lr_beta=5e-4,lr_k3band=1e-4),
}

def now():return datetime.now(timezone.utc)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--init',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--recipe',choices=RATES,required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--sched-total',type=int,default=1000);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--accum',type=int,default=4)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--swa-start',type=int,default=667)
    p.add_argument('--deadline',default=DEADLINE.isoformat());p.add_argument('--rss-cap-gb',type=float,default=72)
    p.add_argument('--groups',default='corrector,v2e,fe31,params,joint')
    p.add_argument('--loss-norm',choices=('net','global'),default='net');p.add_argument('--rate-scale',type=float,default=1.)
    p.add_argument('--d8c-weight',type=float,default=.125);p.add_argument('--without-d8c',action='store_true')
    p.add_argument('--parallel',type=int,choices=(0,2),default=0)
    p.add_argument('--constant-ema',action='store_true',help='Save evaluation-only constant .995 EMA alongside the canonical warm EMA')
    p.add_argument('--literal-frozen-params',action='store_true',help='Emit frozen physics as native literals in a separate learned-only arm')
    p.add_argument('--resume',type=Path)
    p.add_argument('--device-transition',action='store_true',help='Preserve optimizer/order while recording a device or allocation-policy change')
    p.add_argument('--arena-gb',type=float,default=32)
    p.add_argument('--offload-mb',type=float,default=2)
    p.add_argument('--execution-variant',default='paths')
    p.add_argument('--rollback-serial-oom',action='store_true',help='Keep an in-memory full pre-step checkpoint for serial CUDA OOM recovery')
    p.add_argument('--moment-policy-manifest',type=Path,help='General median-window calibration or guard-only declared fork')
    p.add_argument('--leaf-extension-manifest',type=Path,help='Explicit gain-only structural fork with full old-state preservation')
    p.add_argument('--adam-alarm',action='store_true',help='Full-state rollback and stop on nonfinite leaves/moments or >10x one-step v growth')
    p.add_argument('--unwrapped-runtime',action='store_true',help='Retain the original unwrapped source used by the qualified serial/pool engines')
    p.add_argument('--schedule',choices=('cosine','tail','wsd','cooldown'),default='cosine')
    p.add_argument('--tail-constant-steps',type=int,default=800)
    p.add_argument('--continue-sampler',action='store_true',help='Continue the selected endpoint data order with fresh Adam')
    p.add_argument('--fork-optimizer',action='store_true',help='Continue live weights, Adam, EMA, sampler and counter under a new explicit schedule')
    p.add_argument('--fork-rebase-rates',action='store_true',help='Explicitly recompute leaf-scaled base rates on a schedule fork')
    p.add_argument('--constant-ema-decay',type=float,default=.995)
    p.add_argument('--extra-ema-decays',default='',help='Evaluation-only EMA horizons; carried inside every atomic full/recovery checkpoint')
    p.add_argument('--knob-rate-multiplier',type=float,default=1.,help='Explicit one-factor inherited knob-rate fork; neural rates unchanged')
    p.add_argument('--neural-rate-multiplier',type=float,default=1.,help='Inherited non-physics learning-rate factor; knob rates unchanged')
    p.add_argument('--late-swa-window',type=int,default=0)
    p.add_argument('--ml-weight',type=float,default=0.)
    p.add_argument('--independent-auxiliary-manifest',type=Path,
                   help='Frozen TRAIN-only independent MC targets; both matched arms use the same source')
    p.add_argument('--parent-global-step',type=int,help='Reporting offset for a fresh optimizer study from a production endpoint')
    p.add_argument('--noise-aware',action='store_true')
    p.add_argument('--full-only',action='store_true')
    p.add_argument('--d8d-manifest',type=Path)
    p.add_argument('--d8d-weight',type=float,help='Explicit d8d-only loss weight in a qualified owned trainer source')
    p.add_argument('--fork-d8d',action='store_true',help='Full-state data-addition transition')
    p.add_argument('--per-layer-selected',action='store_true')
    p.add_argument('--twin-screen',action='store_true')
    p.add_argument('--screen-every',type=int,default=200)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--ema-decay',type=float,default=.995)
    p.add_argument('--ema-reference-accum',type=int,help='Match EMA warmup and decay in draws to this reference batch')
    p.add_argument('--adam-beta1',type=float)
    p.add_argument('--adam-beta2',type=float)
    p.add_argument('--baseline-cache',type=Path,help='Seed exact frozen-stock MSE cache from an identity-matched donor')
    p.add_argument('--live-baseline-donor',default='',help='Execution-only exact frozen-stock cache feed from an owned supervisor slot')
    p.add_argument('--sampler-seed',type=int,help='Explicit full-state fork of only the sampler seed/order')
    p.add_argument('--skip-resume-initial-read',action='store_true',help='Skip only the redundant readonly32 immediately after exact resume')
    p.add_argument('--skip-fork-initial-read',action='store_true',help='Skip the unchanged parent function read at a declared full-state fork')
    p.add_argument('--audit-updates',type=int,default=0,help='Check this many actual updates against ordinary Adam and named-leaf EMA')
    p.add_argument('--adam-v-reset-manifest',type=Path,help='Frozen2600 second-moment reset plus logged six-sigma guard')
    p.add_argument('--carry-fork-constant-ema',action='store_true')
    p.add_argument('--cuda-topology-proof',type=Path)
    p.add_argument('--full-dataset-path',type=Path)
    p.add_argument('--recovery-every',type=int,default=12)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if a.full_dataset_path:os.environ['WHEST_FULL_DATASET']=str(a.full_dataset_path.resolve())
    if a.fork_optimizer and a.resume:raise ValueError('A schedule fork and exact resume are distinct modes')
    if a.sampler_seed is not None and (a.sampler_seed<0 or not (a.fork_optimizer or a.resume)):
        raise ValueError('Sampler seed variants require an explicit full-state fork or its exact continuation')
    if a.knob_rate_multiplier<=0 or a.knob_rate_multiplier!=1 and not (a.resume or a.fork_optimizer):
        raise ValueError('A changed knob rate requires a positive full-state continuation factor')
    if not 0<a.neural_rate_multiplier<float('inf') or a.neural_rate_multiplier!=1 and not (a.resume or a.fork_optimizer):
        raise ValueError('A changed neural rate requires a positive full-state continuation factor')
    if a.fork_d8d and not (a.fork_optimizer and a.d8d_manifest):
        raise ValueError('A data addition needs an explicit full-state fork and immutable manifest')
    if a.d8d_manifest and a.full_only:raise ValueError('Full-only and added-corpus populations conflict')
    if a.d8d_weight is not None:
        if not a.d8d_manifest or not 0<a.d8d_weight<float('inf'):
            raise ValueError('A d8d-only weight needs a frozen corpus and a finite positive coefficient')
        if 'd8d_loss_weight' not in Path(sys.modules[Engine.__module__].__file__).read_text():
            raise ValueError('This trainer source does not implement the declared d8d-only weight')
    if (a.adam_beta1 is None)!=(a.adam_beta2 is None):raise ValueError('Declare both Adam betas together')
    if a.adam_beta1 is not None and not (0<=a.adam_beta1<1 and 0<=a.adam_beta2<1):
        raise ValueError('Adam betas must lie in [0,1)')
    if a.adam_beta1 is not None and a.fork_optimizer:
        raise ValueError('Draw-time Adam test requires a fresh optimizer, not a schedule fork')
    if a.per_layer_selected and not (a.fork_optimizer or a.resume):
        raise ValueError('Layer-wise capacity starts from a declared full-state fork')
    if a.per_layer_selected and a.full_only:raise ValueError('Capacity and corpus are separate factors')
    if not 0<a.constant_ema_decay<1 or a.late_swa_window<0:raise ValueError('Invalid evaluation average')
    if a.ema_reference_accum is not None and (a.ema_reference_accum<=0 or a.accum%a.ema_reference_accum):
        raise ValueError('Draw-matched EMA requires an integral multiple of its reference batch')
    out=Path(os.environ['SC_SCRATCH'])/'arms'/a.tag;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(33);random.seed(33);np.random.seed(33)
    if a.device=='cuda':
        if os.environ.get('WHEST_CUDA_MEMORY_CAP_GB'):
            torch.cuda.set_per_process_memory_fraction(min(1.0,float(os.environ['WHEST_CUDA_MEMORY_CAP_GB'])*1e9/torch.cuda.get_device_properties(0).total_memory),0)
    twin=RuntimeTwin(a.base,a.device,batch_runtime_validation=not bool(a.parallel or a.unwrapped_runtime))
    twin.model.predict=twin.model._predict_impl
    ck=torch.load(a.resume or a.init,map_location='cpu',weights_only=False)
    if a.cuda_topology_proof:
        from cuda_topology import normalize
        ck,topology=normalize(ck,a.cuda_topology_proof)
        if topology:write_json(out/'cuda-topology-migration.json',topology)
    if not a.resume and not a.fork_optimizer:ck=warm_start_checkpoint(ck,'ema')
    if a.leaf_extension_manifest:
        if not (a.fork_optimizer or a.resume):raise ValueError('Leaf extension requires a fullstate structural fork')
        from leaf_extension import seed
        ck=seed(ck,twin,a.leaf_extension_manifest)
    layer_keys=[]
    inherited_layers=bool(a.per_layer_selected and ck.get('config',{}).get('per_layer_keys'))
    if a.per_layer_selected:
        if a.resume or a.fork_d8d or inherited_layers:
            layer_keys=list(ck['config']['per_layer_keys'])
        else:
            for group in ck['optimizer']['param_groups']:
                name=group['name']
                if not name.startswith('params.') or '.' in name[7:]:continue
                key=name[7:];spec=twin.schema['parameters'][key]
                if spec.get('per_layer_option') and ck['train_state']['params'][key].numel()==1:
                    layer_keys.append(key)
        if not layer_keys:raise ValueError('No trainable shared scalar supports layer-wise expansion')
    continuing=bool(a.resume or a.fork_optimizer)
    # Continuations retain the checkpoint's selected inventory, including any
    # selected knob that reached zero, and do not re-enable initially frozen
    # knobs by re-selecting the entire registry group.
    state=TrainState(twin,ck,warm_start=not continuing,
                     groups=() if continuing else a.groups.split(','),
                     per_layer=layer_keys,freeze_zero_knobs=not continuing)
    state.literal_frozen_params=a.literal_frozen_params
    with torch.no_grad():twin.model.update_runtime(**state.request(evaluation=True))
    state.refresh_schema(twin,include_new_corrector=True)
    population=list(range(800))+data.ranges('0:3072,5120:12536',d8b=True)
    if not a.without_d8c:population+=data.ranges('0:4096',d8c=True)
    d8d_receipt=None
    if a.d8d_manifest:
        from d8d_transition import configure
        added,d8d_receipt=configure(data,a.d8d_manifest)
        population+=added
    if a.full_only:
        population=list(range(800))
        if a.continue_sampler and not (a.fork_optimizer or a.resume):
            raise ValueError('Full-only tail cannot continue a mixed-corpus sampler')
    if a.noise_aware and a.loss_norm!='global':raise ValueError('Noise-aware absolute loss requires global normalization')
    ancestry,known=data.training_ancestry(ck)
    config=dict(population=population,train='0:800',d8b='0:3072,5120:12536',d8c='' if a.without_d8c else '0:4096',
                d8c_loss_weight=a.d8c_weight,seed=33,val='',report_val='',sched_total=a.sched_total,sched_offset=0,
                optimizer_warmup=0 if a.fork_optimizer else a.warmup,ema=a.ema_decay,ema_warmup=1,accum=a.accum,knob_floor=.01,
                offload_mb=a.offload_mb if a.device=='cuda' else 0,offload_arena_gb=a.arena_gb if a.device=='cuda' else 0,
                ml_weight=a.ml_weight,loss_norm=a.loss_norm,s0=1.2e-8,noise_aware=a.noise_aware,
                lr_scale=a.rate_scale,grow_lr=1e-4,
                batched_ema=True,representation='original-DC-polynomial',initial_weights='live-Adam-continuation' if a.fork_optimizer else 'endpoint-ema-current-physics',
                schedule=a.schedule,tail_constant_steps=a.tail_constant_steps,continue_sampler=a.continue_sampler or a.fork_optimizer,
                runtime_config=copy.deepcopy(twin.config),training_ancestry=sorted(set(population+ancestry)),
                holdout_provenance_known=known,groups=a.groups.split(','),literal_frozen_params=a.literal_frozen_params,
                **RATES[a.recipe])
    if a.full_only:
        config.update(d8b='',d8c='',full_only=True,
                      sampler_policy=(ck['config']['sampler_policy'] if a.resume else
                          'fresh seed33 full-only sampler; inherited Adam/EMA on a fork' if a.fork_optimizer else
                          'fresh seed33 full-only sampler; fresh Adam from endpoint EMA'))
    if a.independent_auxiliary_manifest:
        if not a.full_only or a.loss_norm!='global' or a.noise_aware or a.per_layer_selected:
            raise ValueError('Independent auxiliary study requires a plain full-only global pair')
        from tools.train.independent_auxiliary import freeze
        config['independent_auxiliary']=freeze(a.independent_auxiliary_manifest)
    if a.parent_global_step is not None:
        if a.parent_global_step<0 or a.fork_optimizer:raise ValueError('Parent offset is only for a fresh optimizer or its continuation')
        config['parent_global_step']=a.parent_global_step
    if a.adam_v_reset_manifest:
        if not (a.fork_optimizer or a.resume):raise ValueError('Moment intervention requires a declared fork or exact resume')
        from adam_v_reset import descriptor
        config['adam_v_reset']=descriptor(a.adam_v_reset_manifest)
    if a.moment_policy_manifest:
        if not (a.fork_optimizer or a.resume):raise ValueError('Moment policy requires a declared fullstate fork')
        from moment_policy import descriptor
        config['moment_policy']=descriptor(a.moment_policy_manifest)
    if a.adam_alarm:
        if not a.rollback_serial_oom or a.parallel:raise ValueError('Adam alarm requires full serial pre-step rollback state')
        config['adam_alarm']=dict(v_growth_limit=10.,fresh_gain_growth_warmup=20,
            action='rollback and stop; no automatic retrigger loop',nonfinite_guard='Always active, including fresh leaves')
    if a.leaf_extension_manifest:
        from leaf_extension import descriptor
        config['leaf_extension']=descriptor(a.leaf_extension_manifest)
    if d8d_receipt:
        config.update(d8d_manifest=d8d_receipt)
    if a.d8d_weight is not None:
        config['d8d_loss_weight']=a.d8d_weight
    if a.sampler_seed is not None:
        config['sampler_seed_fork']=a.sampler_seed
    if a.ema_reference_accum is not None:
        config['ema_reference_accum']=a.ema_reference_accum
    if a.adam_beta1 is not None:
        config['adam_betas']=[a.adam_beta1,a.adam_beta2]
    if a.knob_rate_multiplier!=1:
        config['knob_rate_multiplier']=a.knob_rate_multiplier
    if a.neural_rate_multiplier!=1:
        config['neural_rate_multiplier']=a.neural_rate_multiplier
    if layer_keys:
        config.update(per_layer_keys=layer_keys,
            per_layer_moment_policy='uniform historical gradient decomposition: m/16,v/256; inherited counters/LR/EMA',
            per_layer_adapter_sha256=hashlib.sha256(Path(__file__).with_name('per_layer_fork.py').read_bytes()).hexdigest())
    if a.resume:
        config['optimizer_warmup']=ck['config'].get('optimizer_warmup',config['optimizer_warmup'])
        config['initial_weights']=ck['config'].get('initial_weights',config['initial_weights'])
    if a.schedule=='tail':
        if not 0<=a.tail_constant_steps<a.sched_total:raise ValueError('Tail constant phase must precede its endpoint')
        import math
        from tools.train import engine as engine_module
        def tail_factor(completed,total,offset=0,*,optimizer_warmup=None):
            warmup=1. if not optimizer_warmup else min(1.,(completed+1)/optimizer_warmup)
            position=max(0.,min(1.,(completed-a.tail_constant_steps)/(total-a.tail_constant_steps)))
            return warmup*(.1+.45*(1+math.cos(math.pi*position)))
        engine_module.schedule_factor=tail_factor
    engine=Engine(twin,state,config)
    if a.adam_beta1 is not None:
        for group in engine.optimizer.param_groups:group['betas']=(a.adam_beta1,a.adam_beta2)
    if a.schedule in ('wsd','cooldown'):
        import math
        from tools.train import engine as engine_module
        def wsd_factor(completed,total,offset=0,*,optimizer_warmup=None):
            warmup=1. if not optimizer_warmup else min(1.,(completed+1)/optimizer_warmup)
            if a.schedule=='wsd':return warmup
            position=max(0.,min(1.,(completed-a.tail_constant_steps)/(total-a.tail_constant_steps)))
            return warmup*(.1+.45*(1+math.cos(math.pi*position)))
        engine_module.schedule_factor=wsd_factor
    if a.continue_sampler and not a.resume and not a.full_only and not a.fork_d8d:
        engine.sampler.load_state_dict(ck['sampler'])
    transition=None
    if a.fork_optimizer:
        if ck['model_identity']!=twin.identity and not a.leaf_extension_manifest:raise ValueError('Moment fork changed source/assets/numerical settings')
        fork_ck=ck
        layer_mapping=[]
        data_transition=None
        if a.leaf_extension_manifest and not ck.get('config',{}).get('leaf_extension'):
            from leaf_extension import extend
            fork_ck,extension_proof=extend(engine,ck,a.leaf_extension_manifest)
            write_json(out/'leaf-extension-proof.json',extension_proof)
            ck=fork_ck
        elif a.fork_d8d:
            from d8d_transition import fork
            fork_ck,data_transition=fork(data,engine,ck)
        elif layer_keys and not inherited_layers:
            from per_layer_fork import expand
            fork_ck,layer_mapping=expand(engine,ck,layer_keys)
        if a.full_only:
            if not set(population)<=set(ck['config']['population']):
                raise ValueError('Full-only fork must use a subset of inherited training rows')
            data.verify_corpus_identity(ck['dataset_identity'])
            old_sources={s['path']:s for s in ck['dataset_identity']['sources']}
            if any(old_sources.get(s['path'])!=s for s in engine.dataset_identity['sources']):
                raise ValueError('Full-only fork changed the inherited full-label files')
            fork_ck=dict(ck,sampler=engine.sampler.state_dict())
        elif not a.fork_d8d and ck['dataset_identity']!=engine.dataset_identity:
            raise ValueError('Moment fork changed data or labels')
        if (not (a.full_only or a.fork_d8d) and ck['config']['population']!=config['population']) or ck['config']['accum']!=a.accum:
            raise ValueError('Moment fork must preserve accumulation and explicitly declare population changes')
        new_groups={g['name']:g['base_lr'] for g in engine.optimizer.param_groups}
        if [g['name'] for g in fork_ck['optimizer']['param_groups']]!=[g['name'] for g in engine.optimizer.param_groups]:
            raise ValueError('Moment fork changed leaf inventory/order')
        engine.restore(fork_ck,exact=False)
        if a.sampler_seed is not None:
            replacement=type(engine.sampler)(population,a.sampler_seed)
            engine.sampler.load_state_dict(replacement.state_dict())
        rate_ratio=a.rate_scale/ck['config'].get('lr_scale',1.)
        knob_ratio=a.knob_rate_multiplier/ck['config'].get('knob_rate_multiplier',1.)
        neural_ratio=a.neural_rate_multiplier/ck['config'].get('neural_rate_multiplier',1.)
        for g in engine.optimizer.param_groups:
            base=new_groups[g['name']] if a.fork_rebase_rates else g['base_lr']*rate_ratio
            if g['name'].startswith(('params.','per_layer.')):base*=knob_ratio
            else:base*=neural_ratio
            g['base_lr']=base;g['lr']=base
        actual_rates={g['name']:g['base_lr'] for g in engine.optimizer.param_groups}
        if a.steps<=engine.completed or a.tail_constant_steps<engine.completed:
            raise ValueError('Moment fork schedule must advance the inherited update counter')
        write_json(out/'fork-start-proof.json',dict(inherited_checkpoint=str(a.init),inherited_completed=engine.completed,
            model_identity_equal=not bool(a.leaf_extension_manifest),dataset_identity_equal=True,leaf_order_equal=not bool(a.leaf_extension_manifest),
            source_extension_manifest=str(a.leaf_extension_manifest) if a.leaf_extension_manifest else None,
            live_weights_adam_ema_rng_preserved=not bool(layer_mapping),
            unexpanded_neural_weights_adam_rng_preserved=True,
            sampler_preserved=not (a.full_only or a.fork_d8d or a.sampler_seed is not None),data_transition=data_transition,
            sampler_seed_fork=a.sampler_seed,
            population_policy='full-only subset, fresh seed33 sampler' if a.full_only else 'inherited population/order',
            inherited_full_label_sources_equal=True,base_rates_replaced=actual_rates,
            per_layer_keys=layer_keys,per_layer_optimizer_mapping=layer_mapping,
            base_rate_policy='explicit-leaf-rebase' if a.fork_rebase_rates else 'inherited-checkpoint-rates',
            inherited_rate_multiplier=rate_ratio,
            inherited_knob_rate_multiplier=knob_ratio,inherited_neural_rate_multiplier=neural_ratio,
            new_recipe=config,timing_collected=False))
    if a.resume:
        if a.device_transition:
            old=copy.deepcopy(ck['model_identity']);new=copy.deepcopy(twin.identity)
            old_settings=old.pop('numerical_settings');new_settings=new.pop('numerical_settings')
            if old!=new or old_settings['dtype']!=new_settings['dtype'] or old_settings['tf32']!=new_settings['tf32']:
                raise ValueError('Device transition requires identical source, assets, configuration, dtype and TF32 policy')
            if ck['dataset_identity']!=engine.dataset_identity:raise ValueError('Device transition changed training labels')
            old_schedule=ck['config'].get('schedule','cosine')
            if old_schedule!=a.schedule or a.schedule in ('tail','cooldown') and ck['config'].get('tail_constant_steps',800)!=a.tail_constant_steps:
                raise ValueError('Device transition changed the scientific schedule; use --fork-optimizer')
            keys=('population','sched_total','sched_offset','optimizer_warmup','ema','ema_warmup','accum','knob_floor',
                  'loss_norm','s0','lr_scale','groups','literal_frozen_params','d8c_loss_weight',*RATES[a.recipe])
            if any(ck['config'].get(k)!=config.get(k) for k in keys):raise ValueError('Device transition changed the scientific recipe')
            engine.restore(ck,exact=False)
            transition=dict(checkpoint=str(a.resume),completed=ck['completed'],from_settings=old_settings,to_settings=new_settings,
                            optimizer_ema_sampler_preserved=True,byte_exact_device_trajectory_claimed=False)
        else:engine.restore(ck)
        if a.adam_beta1 is not None and any(group['betas']!=(a.adam_beta1,a.adam_beta2) for group in engine.optimizer.param_groups):
            raise ValueError('Exact resume changed the declared Adam betas')
    inherited_reset=ck.get('config',{}).get('adam_v_reset')
    if a.adam_v_reset_manifest and inherited_reset and inherited_reset!=config['adam_v_reset']:
        raise ValueError('An optimizer-preserving fork must retain its existing moment intervention descriptor')
    if a.adam_v_reset_manifest and not a.resume and not inherited_reset:
        from adam_v_reset import apply_reset
        reset_proof=apply_reset(engine,a.adam_v_reset_manifest)
        write_json(out/'adam-v-reset-proof.json',reset_proof)
        if reset_proof.get('passed') is not True:raise ValueError('Moment reset changed protected optimizer state')
        fork_proof=json.loads((out/'fork-start-proof.json').read_text())
        fork_proof.update(live_weights_adam_ema_rng_preserved=False,
            intervention='Only selected exp_avg_sq reset; all m/counters/live/EMA/sampler/RNG preserved',
            reset_proof=str(out/'adam-v-reset-proof.json'))
        write_json(out/'fork-start-proof.json',fork_proof)
    inherited_policy=ck.get('config',{}).get('moment_policy')
    if a.moment_policy_manifest:
        if inherited_policy and inherited_policy!=config['moment_policy']:raise ValueError('Moment-policy continuation changed its descriptor')
        if not a.resume and not inherited_policy:
            from moment_policy import apply
            policy_proof=apply(engine,a.moment_policy_manifest)
            write_json(out/'moment-policy-proof.json',policy_proof)
            if policy_proof.get('passed') is not True:raise ValueError('Moment policy changed protected optimizer state')
            fork_proof=json.loads((out/'fork-start-proof.json').read_text())
            fork_proof.update(live_weights_adam_ema_rng_preserved=False,
                intervention='Declared moment-policy recalibration; m/counters/live/EMA/sampler/RNG preserved',
                moment_policy_proof=str(out/'moment-policy-proof.json'))
            write_json(out/'fork-start-proof.json',fork_proof)
    if a.ema_reference_accum is not None:
        from tools.train import engine as engine_module
        original_update_ema=engine_module.update_ema
        virtual_updates=a.accum//a.ema_reference_accum
        reference_decay=a.ema_decay**(1./virtual_updates)
        def matched_update_ema(ema,leaves,decay,*,batched=False):
            matched_decay=1.
            for virtual_step in range(engine.completed*virtual_updates,(engine.completed+1)*virtual_updates):
                matched_decay*=min(reference_decay,(1.+virtual_step)/(10.+virtual_step))
            return original_update_ema(ema,leaves,matched_decay,batched=batched)
        engine_module.update_ema=matched_update_ema
    if layer_keys and not (a.resume or inherited_layers):
        control_state=TrainState(twin,ck,warm_start=False,freeze_zero_knobs=False)
        old_request=control_state.request(evaluation=True);old_request['per_layer']={}
        new_request=state.request(evaluation=True)
        rows=[]
        with torch.no_grad():
            for index in (0,1,2):
                weights,target,noise,metadata=data.net_record(index)
                reference=twin.forward(weights,old_request,seed=metadata['seed']).detach().cpu()
                candidate=twin.forward(weights,new_request,seed=metadata['seed']).detach().cpu()
                delta=candidate.double()-reference.double()
                rel_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference.double()))
                rel_linf=float(delta.abs().max()/reference.double().abs().max())
                rows.append(dict(idx=index,relative_l2=rel_l2,relative_linf=rel_linf,
                                 passed=rel_l2<=2e-6 and rel_linf<=2e-6))
        write_json(out/'capacity-initial-parity.json',dict(rows=rows,keys=layer_keys,
                    passed=all(row['passed'] for row in rows),timing_collected=False))
        if not all(row['passed'] for row in rows):raise ValueError('Per-layer initialization changes the control function')
        twin.model.update_runtime(**state.request(evaluation=True))
        del control_state,old_request,new_request,weights,reference,candidate,delta
        gc.collect()
    cache_seed=None
    live_baseline=None
    if a.live_baseline_donor:
        from live_baseline import LiveBaseline
        live_baseline=LiveBaseline(engine,Path(os.environ['SC_SCRATCH']),a.live_baseline_donor,out)
    if a.baseline_cache is not None:
        from baseline_cache import merge_baseline_cache
        cache_seed=merge_baseline_cache(engine,a.baseline_cache)
        write_json(out/'baseline-cache-seed.json',cache_seed)
    if a.parallel:
        if a.device!='cuda':raise ValueError('Ordered pool requires CUDA')
        if a.literal_frozen_params:raise ValueError('Learned-only literal arm is serial until separately qualified')
        from whest_ts_acceleration.parallel_exact import ExactGradientPool
        os.environ['WHEST_TS_WORKER_HOST_GB']='0'
        engine.gradient_pool=ExactGradientPool(engine,2,a.execution_variant)
    else:
        from tools.train.acceleration_exact import install
        install(engine,a.execution_variant)
    worker_affinity=os.environ.get('WHEST_TO_WORKER_CORES','')
    if worker_affinity:
        children=getattr(getattr(engine,'gradient_pool',None),'children',[])
        allocations=worker_affinity.split(';')
        if len(allocations)!=len(children):raise ValueError('Worker affinity count differs from installed pool')
        for child,cores in zip(children,allocations):
            for task in Path(f'/proc/{child.pid}/task').iterdir():
                os.sched_setaffinity(int(task.name),{int(c) for c in cores.split(',')})
    guard_counts=None
    if a.adam_v_reset_manifest:
        from adam_v_reset import install_guard
        guard_counts=install_guard(engine,out)
    elif a.moment_policy_manifest:
        from moment_policy import install
        guard_counts=install(engine,out)
    manifest=dict(schema='whest-SC-training-arm-v1',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  checkpoint_sha256=hashlib.sha256(a.init.read_bytes()).hexdigest(),source_sha256=twin.source_sha256,
                  model_identity=twin.identity,config=config,n_train=len(population),
                  parameter_count=sum(v.numel() for v in state.leaves.values()),fresh_adam=not bool(a.resume or a.fork_optimizer),device_transition=transition,baseline_cache_seed=cache_seed,
                  guard_disarmed=True,timing_collected=False,started_utc=now().isoformat())
    write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(tag=a.tag,n_train=len(population),parameter_count=manifest['parameter_count'],device=a.device,
                         selected=state.selected,stock_identity=twin.identity)),flush=True)
    stop=False
    def stopping(signum,frame):
        nonlocal stop;stop=True
        print(json.dumps(dict(stop_requested=signum)),flush=True)
    signal.signal(signal.SIGINT,stopping);signal.signal(signal.SIGTERM,stopping)
    swa={};swa_n=0
    if a.resume and engine.completed>=a.swa_start:
        path=a.resume.parent/f'step{engine.completed:04d}-swa.pt'
        if not path.exists():raise ValueError('Resume must retain an already started SWA')
        old_swa=torch.load(path,map_location='cpu',weights_only=False)
        swa_n=old_swa.get('evaluation_average',{}).get('n')
        if swa_n!=engine.completed-a.swa_start+1:raise ValueError('Resumed SWA sample count differs')
        swa={name:value.to(twin.device).clone() for name,value in old_swa['ema'].items()}
    constant_ema={name:value.detach().clone() for name,value in state.leaves.items()} if a.constant_ema else None
    from extra_ema import ExtraEMA
    extra_ema=ExtraEMA(engine,ck if continuing else {},a.extra_ema_decays)
    if constant_ema is not None and (a.resume or a.fork_optimizer and a.carry_fork_constant_ema):
        path=(a.resume or a.init).parent/f'step{engine.completed:04d}-constant-ema.pt'
        if not path.exists():raise ValueError('Resume must retain the original evaluation-only constant EMA')
        old_average=torch.load(path,map_location='cpu',weights_only=False)['ema']
        if a.leaf_extension_manifest and not a.resume and not ck.get('config',{}).get('leaf_extension'):
            from leaf_extension import average
            old_average=average(old_average,state.leaves)
        if old_average.keys()!=constant_ema.keys():raise ValueError('Resumed constant EMA inventory differs')
        constant_ema={name:value.to(twin.device).clone() for name,value in old_average.items()}
    from late_average import LateAverage
    late_saved=None
    if a.resume and a.late_swa_window:
        path=a.resume.parent/f'step{engine.completed:04d}-late-swa{a.late_swa_window}.pt'
        if not path.exists():raise ValueError('Resume must retain the complete late average buffer')
        late_saved=torch.load(path,map_location='cpu',weights_only=False)['evaluation_buffer']
    late_average=LateAverage(a.late_swa_window,late_saved)
    def late_checkpoint():
        averaged=engine.checkpoint()
        averaged['ema']=late_average.mean()
        averaged['evaluation_buffer']=late_average.snapshot()
        averaged['evaluation_average']=dict(method='uniform-late-window-SWA',n=len(late_average),
            requested_window=a.late_swa_window,first_step=engine.completed-len(late_average)+1,last_step=engine.completed,
            accumulation='CPU checkpoint dtype sequential sum with outgoing sample subtraction')
        return averaged
    resume_initial_step=engine.completed if a.resume else None
    fork_initial_step=engine.completed if a.fork_optimizer else None
    def save_endpoint():
        if shutil.disk_usage(out).free<2*10**9:raise OSError('Insufficient free disk for a complete checkpoint')
        path=out/f'step{engine.completed:04d}.pt';engine.save(path)
        with engine.use_weights('ema'):export_state(twin,state,out/f'step{engine.completed:04d}-ema',checkpoint=path)
        for label in extra_ema.buffers:
            averaged=engine.checkpoint();averaged['ema']=cpu_tree(extra_ema.buffers[label])
            averaged['evaluation_average']=dict(method='extra-EMA',decay=float(label),
                start_step=extra_ema.start_step,initialization='canonical EMA at declared horizon start')
            average_path=out/f'step{engine.completed:04d}-ema{label}'
            temporary=out/(average_path.name+'.pt.part');torch.save(averaged,temporary)
            checkpoint_path=out/(average_path.name+'.pt');temporary.replace(checkpoint_path)
            canonical=engine.ema
            try:
                engine.ema=extra_ema.buffers[label]
                with engine.use_weights('ema'):export_state(twin,state,average_path,checkpoint=checkpoint_path)
            finally:engine.ema=canonical
        if swa_n:
            averaged=engine.checkpoint();averaged['ema']=cpu_tree(swa)
            averaged['evaluation_average']=dict(method='SWA',n=swa_n,start_step=a.swa_start)
            torch.save(averaged,out/f'step{engine.completed:04d}-swa.pt')
        if constant_ema is not None:
            averaged=engine.checkpoint();averaged['ema']=cpu_tree(constant_ema)
            averaged['evaluation_average']=dict(method='constant-EMA',decay=a.constant_ema_decay,warmup=False)
            torch.save(averaged,out/f'step{engine.completed:04d}-constant-ema.pt')
        if late_average:
            averaged=late_checkpoint()
            late_path=out/f'step{engine.completed:04d}-late-swa{a.late_swa_window}.pt'
            temporary=late_path.with_suffix('.pt.part');torch.save(averaged,temporary);temporary.replace(late_path)
            canonical=engine.ema
            try:
                engine.ema=averaged['ema']
                with engine.use_weights('ema'):
                    export_state(twin,state,out/f'step{engine.completed:04d}-late-swa{a.late_swa_window}',checkpoint=late_path)
            finally:engine.ema=canonical
        write_json(out/'progress.json',dict(completed=engine.completed,steps_requested=a.steps,swa_n=swa_n,
                   status='running',latest_checkpoint=str(path),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   cuda_peak_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                   observed_training_ids=sorted(set(i for r in engine.history for i in r.get('indices',[r['idx']]))),
                   timing_collected=False))
        print(json.dumps(dict(saved_step=engine.completed,swa_n=swa_n,checkpoint=str(path))),flush=True)
        if a.twin_screen and not (a.skip_resume_initial_read and engine.completed==resume_initial_step) and not (a.skip_fork_initial_read and engine.completed==fork_initial_step):
            val=engine.evaluate(list(range(832,864)),which='ema')
            val.update(schema='whest-TO-twin-val32-v1',tag=a.tag,step=engine.completed,
                       nets_seen=engine.completed*a.accum,baseline='declared-start-source',
                       timing_collected=False,grader_certified=False)
            write_json(out/f'step{engine.completed:04d}-twin-val32.json',val)
            print(json.dumps(dict(twin_val32_step=engine.completed,ratio=val['ratio_noise_corrected'],
                                 wins=val['wins'],z=val['z'])),flush=True)
        if a.parallel:
            try:engine.gradient_pool.trim()
            except (RuntimeError,OSError,EOFError) as error:
                print(json.dumps(dict(worker_trim_failed_after_checkpoint=str(error))),flush=True)
    def save_backup():
        if shutil.disk_usage(out).free<2*10**9:raise OSError('Insufficient free disk for recovery checkpoint')
        # Recovery checkpoints are not candidate exports or extra selection points.
        folder=out/'recovery';folder.mkdir(exist_ok=True)
        path=folder/f'step{engine.completed:04d}.pt';engine.save(path)
        if constant_ema is not None:
            averaged=engine.checkpoint();averaged['ema']=cpu_tree(constant_ema)
            averaged['evaluation_average']=dict(method='constant-EMA',decay=a.constant_ema_decay,warmup=False)
            temporary=folder/f'step{engine.completed:04d}-constant-ema.pt.part'
            torch.save(averaged,temporary);temporary.replace(temporary.with_suffix(''))
        if swa_n:
            averaged=engine.checkpoint();averaged['ema']=cpu_tree(swa)
            averaged['evaluation_average']=dict(method='SWA',n=swa_n)
            temporary=folder/f'step{engine.completed:04d}-swa.pt.part'
            torch.save(averaged,temporary);temporary.replace(temporary.with_suffix(''))
        if late_average:
            temporary=folder/f'step{engine.completed:04d}-late-swa{a.late_swa_window}.pt.part'
            torch.save(late_checkpoint(),temporary);temporary.replace(temporary.with_suffix(''))
        write_json(folder/'ready.json',dict(completed=engine.completed,checkpoint=str(path),swa_n=swa_n))
    save_endpoint()
    try:
        for index in range(max(0,a.steps-engine.completed)):
            if stop or now()>=deadline:break
            if live_baseline is not None and index%6==0:live_baseline.pull()
            sampler_before=engine.sampler.state_dict() if a.parallel else None
            serial_before=engine.checkpoint() if a.rollback_serial_oom and not a.parallel else None
            serial_leaves=cpu_tree(state.leaves) if serial_before is not None else None
            completed_before=engine.completed
            update_audit=None
            if index<a.audit_updates:
                from optimizer_ema_audit import UpdateAudit
                update_audit=UpdateAudit(engine)
            try:row=engine.step()
            except (RuntimeError,FloatingPointError) as error:
                if a.adam_alarm and (isinstance(error,FloatingPointError) or 'nonfinite' in str(error).lower()):
                    engine.restore(serial_before)
                    with torch.no_grad():
                        for name,value in state.leaves.items():value.copy_(serial_leaves[name].to(value))
                    torch.save(serial_before,out/'alarm-prestep.pt')
                    slot=os.environ.get('WHEST_TO_SLOT')
                    if slot:write_json(Path(os.environ['TO_SCRATCH'])/'receipts'/('STOP-'+slot),dict(reason='Nonfinite Adam alarm requires review',error=str(error),checkpoint=str(out/'alarm-prestep.pt')))
                    write_json(out/'adam-alarm.json',dict(restored_step=engine.completed,error=str(error),
                        full_pre_step_state_restored=True,live_leaves_restored_in_place=True,
                        extra_ema_constant_ema_not_yet_updated=True,automatic_restart_disabled=bool(slot),timing_collected=False))
                    stop=True;break
                if serial_before is not None and 'CUDA out of memory' in str(error):
                    engine.restore(serial_before)
                    # Engine.restore restores optimizer/EMA/order but its state
                    # object keeps the current leaves. Restore them explicitly
                    # in place, retaining the optimizer's parameter references.
                    with torch.no_grad():
                        for name,value in state.leaves.items():value.copy_(serial_leaves[name].to(value))
                    write_json(out/'worker-failure.json',dict(completed=engine.completed,
                         error=str(error),full_pre_step_checkpoint_restored=True,
                         live_leaves_restored_in_place=True,
                         sampler_rolled_back=True,optimizer_update_retained=False))
                    stop=True;break
                if (a.parallel and engine.completed==completed_before
                    and 'CUDA out of memory' in str(error) and 'parallel_exact.py' in str(error)):
                    # Pool errors precede canonical gradient addition/Adam/EMA.
                    engine.sampler.load_state_dict(sampler_before)
                    write_json(out/'worker-failure.json',dict(completed=engine.completed,
                         error=str(error),sampler_rolled_back=True,optimizer_update_applied=False))
                    stop=True;break
                raise
            if a.adam_alarm:
                from moment_policy import anomalies
                events=anomalies(engine,serial_before)
                if events:
                    rejected_step=engine.completed;engine.restore(serial_before)
                    with torch.no_grad():
                        for name,value in state.leaves.items():value.copy_(serial_leaves[name].to(value))
                    torch.save(serial_before,out/'alarm-prestep.pt')
                    slot=os.environ.get('WHEST_TO_SLOT')
                    if slot:write_json(Path(os.environ['TO_SCRATCH'])/'receipts'/('STOP-'+slot),dict(reason='Adam alarm requires review',events=events,checkpoint=str(out/'alarm-prestep.pt')))
                    write_json(out/'adam-alarm.json',dict(rejected_step=rejected_step,restored_step=engine.completed,
                        events=events,full_pre_step_state_restored=True,live_leaves_restored_in_place=True,
                        extra_ema_constant_ema_not_yet_updated=True,automatic_restart_disabled=bool(slot),timing_collected=False))
                    stop=True;break
            if update_audit is not None:
                audit=update_audit.check(engine)
                write_json(out/f'optimizer-ema-audit-step{engine.completed:04d}.json',audit)
                if not audit['passed']:raise ValueError('Optimizer/EMA update audit failed; bank remains blocked')
                update_audit=None
            serial_before=None
            serial_leaves=None
            extra_ema.update(state.leaves)
            if constant_ema is not None:
                with torch.no_grad():
                    alpha=.005 if a.constant_ema_decay==.995 else 1-a.constant_ema_decay
                    for name,value in state.leaves.items():constant_ema[name].mul_(a.constant_ema_decay).add_(value.detach(),alpha=alpha)
            if a.late_swa_window:
                late_average.add(state.leaves)
            if engine.completed>=a.swa_start:
                with torch.no_grad():
                    if not swa:swa={name:value.detach().clone() for name,value in state.leaves.items()}
                    else:
                        for name,value in state.leaves.items():swa[name].add_((value.detach()-swa[name])/(swa_n+1))
                    swa_n+=1
            if index%5==0:
                print(json.dumps(row),flush=True)
                write_json(out/'heartbeat.json',dict(step=engine.completed,loss=row['loss'],ids=row.get('indices',[row['idx']]),
                    current_rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                    timing_collected=False))
                if guard_counts is not None:write_json(out/'gradient-guard-summary.json',dict(guard_counts,timing_collected=False))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>a.rss_cap_gb*1e9:
                stop=True;print(json.dumps(dict(resource_stop='RSS ceiling',step=engine.completed)),flush=True)
            if engine.completed%a.save_every==0:save_endpoint()
            if engine.completed%a.recovery_every==0:save_backup()
            # Explicit collections avoid retained cyclic graphs over changing rows.
            gc.collect()
    finally:
        if guard_counts is not None:write_json(out/'gradient-guard-summary.json',dict(guard_counts,timing_collected=False))
        if engine.completed%a.save_every:save_endpoint()
        write_json(out/'completion.json',dict(completed=engine.completed,steps_requested=a.steps,
                   cosine_completed=engine.completed>=a.sched_total,
                   status='complete' if engine.completed==a.steps else 'stopped-at-update-boundary',swa_n=swa_n,
                   stopped_utc=now().isoformat(),timing_collected=False))
        engine.save(out/'continuation.pt')
        if hasattr(engine,'acceleration_stats'):
            write_json(out/'execution-counts.json',{k:dict(v) for k,v in engine.acceleration_stats.items()})
            engine.close_acceleration()
        if a.parallel:
            write_json(out/'worker-counts.json',dict(engine.gradient_pool.stats))
            engine.gradient_pool.close()
    print(json.dumps(dict(tag=a.tag,completed=engine.completed,finished=True)),flush=True)

if __name__=='__main__':main()
