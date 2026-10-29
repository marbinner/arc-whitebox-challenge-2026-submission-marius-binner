"""Select checkpoint model weights without restoring a training session."""
from contextlib import contextmanager

import torch


def warm_start_checkpoint(checkpoint, which='live'):
    """Materialize EMA before new-source policy, selection, growth or zero freezing.

    Preserve the original checkpoint and ancestry. Optimizer, schedule and sampler
    remain warm-start metadata and are not restored by --init.
    """
    if which not in ('live', 'ema'):
        raise ValueError('Initial weights must be live or ema')
    if which == 'live':
        return checkpoint
    if (checkpoint is None or checkpoint.get('schema') != 'whest-train-v1'
            or checkpoint.get('config', {}).get('ema', 0) <= 0
            or not checkpoint.get('ema')):
        raise ValueError('--init-weights ema requires a whest-train-v1 checkpoint with EMA')
    import copy
    result = dict(checkpoint)
    snapshot = copy.deepcopy(checkpoint['train_state'])
    result['train_state'] = snapshot
    groups = checkpoint.get('optimizer', {}).get('param_groups', [])
    names = [group.get('name') for group in groups]
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError('EMA initialization requires a named optimizer leaf inventory')
    if len(set(names)) != len(names) or set(names) != set(checkpoint['ema']):
        raise ValueError('Checkpoint EMA does not match its original optimizer leaves')

    def key(mapping, part):
        if part in mapping:
            return part
        if part.isdecimal() and int(part) in mapping:
            return int(part)
        raise ValueError('EMA tensor has no checkpoint leaf: ' + part)

    for name, value in checkpoint['ema'].items():
        parts = name.split('.')
        prefix = parts.pop(0)
        if prefix in ('corr', 'v2e'):
            tree = snapshot['theta' if prefix == 'corr' else 'v2e']
            leaf_key = '.'.join(parts)
        elif prefix == 'fe31' and parts == ['wnew']:
            tree, leaf_key = snapshot, 'wnew'
        elif prefix in ('params', 'per_layer'):
            tree = snapshot['params' if prefix == 'params' else 'schedules']
            while len(parts) > 1:
                tree = tree[key(tree, parts.pop(0))]
            leaf_key = key(tree, parts[0])
        else:
            tree = snapshot.get('extra_assets', {}).get(prefix, {})
            leaf_key = '.'.join(parts)
        if (leaf_key not in tree or not isinstance(tree[leaf_key], torch.Tensor)
                or not isinstance(value, torch.Tensor) or value.shape != tree[leaf_key].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid initial EMA tensor: ' + name)
        tree[leaf_key] = value.detach().to(device=tree[leaf_key].device,
                                          dtype=tree[leaf_key].dtype).clone()
    result['warm_start_weights'] = which
    return result


class CheckpointWeights:
    def __init__(self, state, checkpoint):
        self.state = state
        self.decay = checkpoint.get('config', {}).get('ema', 0)
        self.ema = checkpoint.get('ema', {})
        if self.decay > 0:
            if set(self.ema) != set(state.leaves):
                raise ValueError('Checkpoint EMA does not match model leaves')
            for name, value in state.leaves.items():
                if self.ema[name].shape != value.shape or not bool(torch.isfinite(self.ema[name]).all()):
                    raise ValueError('Invalid checkpoint EMA tensor: ' + name)

    @contextmanager
    def use_weights(self, which='ema'):
        if which not in ('live', 'ema'):
            raise ValueError('Weights must be live or ema')
        if which == 'live' or self.decay <= 0:
            yield
            return
        previous = {name: value.detach().clone() for name, value in self.state.leaves.items()}
        try:
            with torch.no_grad():
                for name, value in self.state.leaves.items():
                    value.copy_(self.ema[name])
            yield
        finally:
            with torch.no_grad():
                for name, value in self.state.leaves.items():
                    value.copy_(previous[name])
