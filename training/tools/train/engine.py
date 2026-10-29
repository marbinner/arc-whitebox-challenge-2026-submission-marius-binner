"""f3 loss/Adam/EMA recipe, with reproducible continuation checkpoints."""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch

from . import data
from .runtime import RuntimeTwin, cpu_tree
from .state import TrainState
from .checkpoint import CheckpointWeights
from .identity import recipe_identity, loader_identity


class Sampler:
    def __init__(self, population, seed=33):
        if not population: raise ValueError('The training panel is empty')
        self.population = list(population)
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor = self.rng.permutation(self.population).tolist(), 0

    def next(self):
        if self.cursor == len(self.order):
            self.order, self.cursor = self.rng.permutation(self.population).tolist(), 0
        item = int(self.order[self.cursor]); self.cursor += 1
        return item

    def state_dict(self):
        return dict(population=list(self.population), order=list(self.order), cursor=self.cursor,
                    rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, value):
        if self.population != value['population']:
            raise ValueError('Resume panel differs from checkpoint; use --init for a new recipe')
        self.order, self.cursor = list(value['order']), int(value['cursor'])
        self.rng.bit_generator.state = copy.deepcopy(value['rng'])


def schedule_factor(completed, total, offset=0):
    if not math.isfinite(total) or total<=0: raise ValueError('Schedule total must be finite and positive')
    if not math.isfinite(offset) or offset<0: raise ValueError('Schedule offset must be finite and nonnegative')
    if not math.isfinite(completed) or completed<0: raise ValueError('Schedule position must be finite and nonnegative')
    position = completed + offset
    return min(1.0, (position + 1) / 50) * (0.1 + 0.45 *
               (1 + math.cos(math.pi * min(position, total) / total)))


def validate_recipe(config):
    schedule_factor(0,config.get('sched_total',3000),config.get('sched_offset',0))
    if type(config.get('accum',1)) is not int or config.get('accum',1)<1:
        raise ValueError('accum must be a positive integer')
    if not config.get('knob_floor',0.3)>0:
        raise ValueError('knob-floor must be positive')
    for key,value in config.items():
        if key.startswith('lr_') and key!='lr_overrides' or key in ('grow_lr','ema','s0','offload_mb','offload_arena_gb','ml_weight','knob_floor'):
            if not math.isfinite(value): raise ValueError('Recipe value must be finite: '+key)


def nonfinite_gradient_names(gradients):
    """One host decision per device; transfer individual flags only on failure."""
    by_device = {}
    for name, value in gradients.items():
        if value is not None:
            by_device.setdefault(value.device, []).append((name, value))
    bad = set()
    for values in by_device.values():
        if len(values) == 1:
            name, value = values[0]
            if not bool(torch.isfinite(value).all()):
                bad.add(name)
            continue
        flags = torch.stack([torch.isfinite(value).all() for _, value in values])
        if not bool(flags.all()):
            bad.update(name for (name, _), finite in zip(values, flags.tolist()) if not finite)
    return [name for name in gradients if name in bad]


class Engine:
    def __init__(self, twin, state, config, *, loader=data.net, labels=data.labels_all):
        self.twin, self.state, self.config = twin, state, copy.deepcopy(config)
        inherited=getattr(state,'training_provenance',data.normalize_provenance(None))
        self.config['training_ancestry']=sorted(set(self.config.get('training_ancestry',config['population'])+inherited['seen_indices']))
        self.config['holdout_provenance_known']=self.config.get('holdout_provenance_known',True) and inherited['known']
        self.config.setdefault('ancestry_corpus_sha256',[])
        validate_recipe(self.config)
        self.loader, self.labels = loader, labels
        held_out = data.panel(config.get('val','832:864')) + data.panel(config.get('report_val',''))
        data.validate_training_panel(self.config['training_ancestry'],held_out,
                                     in_sample=config.get('in_sample',False))
        self.recipe_identity = recipe_identity(self.config)
        self.dataset_identity = (data.corpus_identity(config['population']+held_out)
                                 if loader is data.net else loader_identity(loader))
        if any(shape is not None for shape in state.grown.values()) and config.get('grow_lr',1e-4)<=0:
            raise ValueError('grow-lr must be positive when widening existing slices')
        self.sampler = Sampler(config['population'], config.get('seed', 33))
        rates = {key: config[key] for key in ('lr_corr','lr_v2e','lr_knobs','lr_beta',
                    'lr_fe31','lr_k3band','lr_joint','lr_scale','grow_lr','knob_floor') if key in config}
        groups = state.optimizer_groups(**rates, overrides=config.get('lr_overrides'))
        self.optimizer = torch.optim.Adam(groups, foreach=False)
        self.ema = {name: value.detach().clone() for name, value in state.leaves.items()}
        self.completed, self.history = 0, []
        self.base_cache, self.layer_cache, self.stock = {}, {}, None
        self.best = None
        self.best_artifact = None

    def load_row(self, idx):
        if self.loader is data.net:return data.net_record(idx)
        row=self.loader(idx)
        return (*row,{}) if len(row)==3 else row

    def stock_prediction(self, weights, *, seed=None):
        if self.stock is None:
            stock = RuntimeTwin(self.twin.base, self.twin.device,config=self.twin.config,
                                **({'packed_checkpoint':True} if getattr(self.twin,'packed_checkpoint',False) else {}))
            expected = getattr(self.twin, 'identity', None)
            if expected is not None and getattr(stock, 'identity', None) != expected:
                raise ValueError('Stock baseline source, assets or runtime identity changed during training')
            self.stock = stock
        with torch.no_grad():
            return self.stock.forward(weights,**({'seed':seed} if seed is not None else {})).double().cpu()

    def base_mse(self, idx, weights, target, *, seed=None, targets=None):
        if idx not in self.base_cache:
            pred = self.stock_prediction(weights,seed=seed)
            self.base_cache[idx] = float(((pred[-1] - target.cpu()) ** 2).mean())
            if targets is not None and idx not in self.layer_cache:
                self.layer_cache[idx] = ((pred - targets.cpu()) ** 2).mean(dim=1).tolist()
        return self.base_cache[idx]

    def layer_mse(self, idx, weights, targets, *, seed=None):
        if idx not in self.layer_cache:
            pred = self.stock_prediction(weights,seed=seed)
            self.layer_cache[idx] = ((pred - targets.cpu()) ** 2).mean(dim=1).tolist()
        return self.layer_cache[idx]

    def objective(self, idx, weights, target, *, evaluation=False, seed=None):
        targets = self.labels(idx) if self.config.get('ml_weight', 0) > 0 and idx >= 0 else None
        # On a fresh full row, the same stock prediction supplies every denominator.
        # The default final-only path has exactly the original operations and calls.
        scale = self.config.get('s0', 1.2e-8) if self.config.get('loss_norm', 'net') == 'global' else self.base_mse(
            idx, weights, target, seed=seed, **({'targets': targets} if targets is not None else {}))
        pred = self.twin.forward(weights, self.state.request(evaluation=evaluation),
                                 offload_mb=0 if evaluation else self.config.get('offload_mb', 0),
                                 **({'offload_arena_gb':self.config['offload_arena_gb']}
                                    if self.config.get('offload_arena_gb',0)>0 and not evaluation else {}),
                                 **({'seed':seed} if seed is not None else {}))
        loss = ((pred[-1].double() - target.to(pred.device)) ** 2).mean() / max(scale, 1e-30)
        if targets is not None and pred.shape[0] > 2:
            scales = self.layer_mse(idx, weights, targets,**({'seed':seed} if seed is not None else {}))
            terms = [((pred[layer].double() - targets[layer].to(pred.device)) ** 2).mean() /
                     max(scales[layer], 1e-30) for layer in range(1, pred.shape[0] - 1)]
            loss = loss + self.config['ml_weight'] * torch.stack(terms).mean()
        return loss

    def step(self):
        data.verify_corpus_identity(self.dataset_identity)
        idx = self.sampler.next()
        factor = schedule_factor(self.completed, self.config['sched_total'], self.config.get('sched_offset', 0))
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        # Gradient accumulation (DC's refit recipe ran accum 4): one Adam update from the mean loss of `accum`
        # consecutive sampler networks. accum 1 is the original single-network step, operation for operation.
        accum = self.config.get('accum', 1)
        indices = [idx] + [self.sampler.next() for _ in range(accum - 1)]
        losses = []
        for position, net in enumerate(indices):
            weights, target, _, metadata = self.load_row(net)
            if not position:
                self.optimizer.zero_grad(set_to_none=True)
            term = self.objective(net, weights, target,seed=metadata.get('seed'))
            if not bool(torch.isfinite(term)):
                raise FloatingPointError('Nonfinite loss on training net ' + str(net))
            (term / accum if accum > 1 else term).backward()
            losses.append(term.detach())
            del term
        loss = losses[0] if accum == 1 else torch.stack(losses).mean()
        gradients = {name: value.grad for name, value in self.state.leaves.items()}
        bad = nonfinite_gradient_names(gradients)
        if bad: raise FloatingPointError('Nonfinite gradients: ' + ', '.join(bad))
        if all(value is None for value in gradients.values()):
            raise RuntimeError('None of the selected trainables receives a gradient')
        previous = {}
        for key, shape in self.state.grown.items():
            if shape is not None and 'corr.' + key in self.state.leaves:
                slices = tuple(slice(0, size) for size in shape)
                previous[key] = (slices, self.state.theta[key].detach()[slices].clone())
        self.optimizer.step()
        with torch.no_grad():
            groups={group['name']:group for group in self.optimizer.param_groups}
            for key, (slices, old) in previous.items():
                value = self.state.theta[key]
                group=groups.get('corr.'+key,groups.get('corrector.'+key))
                if group is None: raise ValueError('Grown tensor has no resolved optimizer group: '+key)
                ratio=group['old_slice_ratio']
                value[slices] = old + (value[slices] - old) * ratio
            self.state.clamp_parameters()
            decay = self.config.get('ema', 0.995)
            if self.config.get('ema_warmup', 0):
                # Warm-up EMA: a fixed 0.995 EMA started at the init captures only 44 % of a 200-step
                # refit's movement (68 % at 400); this captures ~97 % and is unchanged for long runs.
                decay = min(decay, (1.0 + self.completed) / (10.0 + self.completed))
            for name, value in self.state.leaves.items():
                self.ema[name].mul_(decay).add_(value.detach(), alpha=1.0-decay)
        self.completed += 1
        row = dict(step=self.completed, idx=idx, loss=float(loss.detach()), schedule_factor=factor,
                   no_gradient=[name for name,value in gradients.items() if value is None])
        if accum > 1: row['indices'] = indices
        self.history.append(row)
        return row

    @contextmanager
    def use_weights(self, which='ema'):
        with CheckpointWeights(self.state,dict(config=self.config,ema=self.ema)).use_weights(which):
            yield

    def evaluate(self, panel, which='ema'):
        panel = data.validate_evaluation_panel(panel)
        data.verify_corpus_identity(self.dataset_identity)
        rows = []
        with self.use_weights(which), torch.no_grad():
            for idx in panel:
                weights, target, noise, metadata = self.load_row(idx)
                seed=metadata.get('seed')
                pred = self.twin.forward(weights, self.state.request(evaluation=True),**({'seed':seed} if seed is not None else {}))
                if pred.ndim!=2 or tuple(pred[-1].shape)!=tuple(target.shape) or not bool(torch.isfinite(pred).all()):
                    raise ValueError('Nonfinite or invalid evaluation prediction on net '+str(idx))
                mse = float(((pred[-1].double().cpu() - target.cpu()) ** 2).mean())
                baseline=self.base_mse(idx,weights,target,seed=seed)
                if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in (mse,baseline,noise)):
                    raise ValueError('Nonfinite or invalid evaluation MSE/noise on net '+str(idx))
                rows.append(dict(metadata,idx=idx, full=mse, base=baseline, noise=noise))
        if not rows: return dict(rows=[],n=0)
        a,b = np.array([r['full'] for r in rows]), np.array([r['base'] for r in rows])
        noise = np.array([r['noise'] for r in rows])
        gains = 1.0 - a / b if (b > 0).all() else None
        z = (float(gains.mean() / (gains.std(ddof=1) / math.sqrt(len(rows))))
             if gains is not None and len(rows) > 1 and gains.std(ddof=1) > 0 else None)
        raw_denominator, corrected_denominator = b.sum(), (b - noise).sum()
        return dict(n=len(rows), mse=float(a.mean()), wins=int((a<b).sum()),z=z,
                    ratio_vs_stock=float(a.sum()/raw_denominator) if raw_denominator > 0 else None,
                    ratio_noise_corrected=float((a-noise).sum()/corrected_denominator)
                        if corrected_denominator > 0 else None, rows=rows)

    def checkpoint(self):
        return dict(schema='whest-train-v1',train_state=self.state.snapshot(),
                    optimizer=cpu_tree(self.optimizer.state_dict()), ema=cpu_tree(self.ema),
                    completed=self.completed,sampler=self.sampler.state_dict(),config=copy.deepcopy(self.config),
                    history=copy.deepcopy(self.history),base_cache=dict(self.base_cache),layer_cache=copy.deepcopy(self.layer_cache),
                    best=self.best,best_artifact=copy.deepcopy(self.best_artifact),source_sha256=self.twin.source_sha256,
                    training_provenance=dict(seen_indices=list(self.config['training_ancestry']),
                        known=self.config['holdout_provenance_known'],
                        corpus_sha256=sorted(set(self.config.get('ancestry_corpus_sha256',[])+[
                            self.dataset_identity.get('labels_sha256',self.dataset_identity.get('sha256','unknown'))]+
                            getattr(self.state,'training_provenance',{}).get('corpus_sha256',[])))),
                    model_identity=copy.deepcopy(getattr(self.twin,'identity',None)),
                    recipe_identity=copy.deepcopy(self.recipe_identity),dataset_identity=copy.deepcopy(self.dataset_identity),
                    rng=dict(torch=torch.get_rng_state(),python=random.getstate(),
                             numpy=np.random.get_state(),cuda=torch.cuda.get_rng_state_all()
                             if self.twin.device.type=='cuda' else None))

    def restore(self, checkpoint,*,exact=True):
        if checkpoint.get('schema') != 'whest-train-v1':
            raise ValueError('Exact resume needs a whest-train-v1 checkpoint; use --init for legacy files')
        if exact and checkpoint['source_sha256'] != self.twin.source_sha256:
            raise ValueError('Exact resume requires the same base source; use --init after structural changes')
        if exact and 'model_identity' not in checkpoint:
            raise ValueError('Checkpoint lacks execution identities; use --init as a warm start')
        if exact and checkpoint['model_identity']!=getattr(self.twin,'identity',None):
            raise ValueError('Exact resume requires the same assets and runtime configuration; use --init')
        if exact and checkpoint.get('recipe_identity') != self.recipe_identity:
            raise ValueError('Exact resume requires the same trainer recipe identity; use --init')
        if exact and checkpoint.get('dataset_identity') != self.dataset_identity:
            raise ValueError('Exact resume requires the same dataset/label identity; use --init')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema = {name:value.to(self.twin.device).clone() for name,value in checkpoint['ema'].items()}
        self.completed,self.history,self.best = checkpoint['completed'],copy.deepcopy(checkpoint['history']),checkpoint.get('best')
        self.best_artifact=copy.deepcopy(checkpoint.get('best_artifact'))
        self.sampler.load_state_dict(checkpoint['sampler'])
        self.base_cache,self.layer_cache = dict(checkpoint['base_cache']),copy.deepcopy(checkpoint['layer_cache'])
        rng=checkpoint['rng']
        torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        if rng['cuda'] is not None and self.twin.device.type=='cuda': torch.cuda.set_rng_state_all(rng['cuda'])

    def save(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.part')
        torch.save(self.checkpoint(),temporary);temporary.replace(path)
