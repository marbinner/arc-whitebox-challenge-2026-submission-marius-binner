"""Training/eval networks for the base corrector trainer (f3 2026-10-07).
idx ≥ 0: aicrowd/arc-whestbench-public-2026 split `full` (revision v2-phase2; local HF cache), all-layer N = 1e9 means.
idx < 0: d8b corpus net i = −idx − 1 (keenanpepper/arc-whestbench-p2-d8b-corpus-14k): weights from torch CPU generator seed 770000 + i,
         final-layer N = 1e8 mean; 'noise' = mean_j Var(h16_j)/N (exact label-noise term of the observed MSE)."""
import os
from functools import lru_cache
import re
import hashlib
from pathlib import Path

from .. import data as canonical

import numpy as np
import torch
from ..export_manifest import normalize_provenance,merge_provenance,package_provenance

D8B = canonical.D8B
WIDTH, DEPTH = 1024, 16

# Keep all legacy full/d8b integer identities unchanged. d8c occupies a new
# negative namespace, including checkpoint/provenance seen_indices.
D8C = canonical.D8C
D8C_OFFSET = 1_000_000


def d8c_identity(index):
    if type(index) is not int or not 0<=index<canonical.D8C_INDEX_BOUND:
        raise ValueError('d8c index is outside its dataset')
    return -(D8C_OFFSET+index+1)


def d8c_weights(index):
    generator=torch.Generator(device='cpu').manual_seed(canonical.D8C_W_SEED0+int(index))
    return torch.randn(DEPTH,WIDTH,WIDTH,generator=generator,dtype=torch.float32)*(2.0/WIDTH)**0.5




def reserved_panels(path=None):
    """Use the repository's panel authority, including added one-time d8b panels."""
    path = Path(path) if path else Path(__file__).resolve().parents[2] / 'panels/PANELS.md'
    panels = {}
    for line in path.read_text().splitlines():
        columns = [part.strip() for part in line.strip().strip('|').split('|')]
        if len(columns) < 2:
            continue
        name, text = columns[:2]
        if name not in ('VAL', 'TEST', 'FINAL104') and not name.startswith('D8B-'):
            continue
        specs = re.findall(r'\d+:\d+', text)
        if not specs:
            raise ValueError('Reserved panel has no network range: ' + name)
        panels[name] = set(ranges(','.join(specs), d8b=text.startswith('d8b ')))
    if not {'VAL', 'TEST', 'FINAL104', 'D8B-A', 'D8B-B', 'D8B-C'} <= panels.keys():
        raise ValueError('Reserved panel definitions are incomplete')
    return panels


def validate_evaluation_panel(panel):
    """One observation per network; repetition is a training-only weighting choice."""
    panel = list(panel)
    if any(type(index) is not int for index in panel):
        raise ValueError('Evaluation panel identities must be integers')
    if len(set(panel)) != len(panel):
        raise ValueError('Duplicate networks in evaluation panel')
    return panel


def validate_training_panel(population, validation=(), *, in_sample=False):
    validation = validate_evaluation_panel(validation)
    if in_sample:
        return
    training = set(population)
    panels = reserved_panels()
    panels['configured validation'] = set(validation)
    overlap = {name: sorted(training.intersection(rows)) for name, rows in panels.items()
               if training.intersection(rows)}
    if overlap:
        raise ValueError('Training overlaps held-out/reserved panels: ' + repr(overlap) +
                         '; use --in-sample to label this experiment explicitly')


def training_ancestry(checkpoint):
    if checkpoint is None: return [],True
    provenance=checkpoint.get('training_provenance',{})
    population=provenance.get('seen_indices',checkpoint.get('config',{}).get('population',[]))
    if any(type(index) is not int for index in population): raise ValueError('Invalid training ancestry identities')
    known=provenance.get('known',bool(population and checkpoint.get('dataset_identity')))
    return sorted(set(population)),known


def checkpoint_provenance(checkpoint):
    if checkpoint is None:return normalize_provenance(None)
    population,known=training_ancestry(checkpoint)
    return normalize_provenance(dict(seen_indices=population,known=known,
        corpus_sha256=checkpoint.get('training_provenance',{}).get('corpus_sha256',[])))


def _full_label_rows(dataset, indices):
    """Hash selected prepared labels with one Arrow open per touched shard.

    Freeze the prepared index across this batch; the caller separately verifies
    every frozen source version and content fingerprint before accepting it.
    """
    if not isinstance(dataset, canonical.PreparedRows):
        labels = dataset.select_columns(['all_layer_means', 'mlp_seed', 'avg_variance']) if hasattr(dataset, 'select_columns') else dataset
        for idx in indices:
            yield idx, labels.labels(idx) if hasattr(labels, 'labels') else labels[idx]
        return
    from datasets import Dataset
    dataset._refresh()
    fingerprint, index = dataset._fingerprint, dataset.index
    if any(type(idx) is not int or not 0 <= idx < index['rows'] for idx in indices):
        raise IndexError('Full label index outside prepared dataset')
    cursor = 0
    for shard in index['shards']:
        first = cursor
        while cursor < len(indices) and indices[cursor] < shard['stop']:
            cursor += 1
        if cursor == first:
            continue
        labels = Dataset.from_file(str(dataset.directory / shard['filename']))
        labels = labels.select_columns(['all_layer_means', 'mlp_seed', 'avg_variance']).with_format('numpy')
        if len(labels) != shard['stop'] - shard['start']:
            raise ValueError('Prepared label shard changed during fingerprinting')
        for idx in indices[first:cursor]:
            yield idx, labels[idx - shard['start']]
    dataset._refresh()
    if dataset._fingerprint != fingerprint:
        raise ValueError('Prepared label sources changed during fingerprinting')


def corpus_identity(indices):
    """Hash the labels actually used, without materializing MLP weight columns."""
    indices = sorted(set(indices))
    value = hashlib.sha256()
    sources = []
    full = [idx for idx in indices if idx >= 0]
    if full:
        dataset = _full()
        paths=(dataset.cache_files if hasattr(dataset,'cache_files') else
               [dict(filename=dataset.directory/'state.json')]+
               [dict(filename=dataset.directory/shard['filename']) for shard in dataset.index['shards']])
        for item in paths:
            path=Path(item['filename']).resolve();stat=path.stat()
            sources.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,head_tail_sha256=_head_tail_sha256(path)))
        for idx,row in _full_label_rows(dataset, full):
            value.update(str(idx).encode()+b'\0'+str(int(row['mlp_seed'])).encode())
            value.update(np.asarray(row['all_layer_means'],dtype=np.float64).tobytes())
            value.update(np.float64(row['avg_variance']/1e9).tobytes())
    d8b = sorted(-idx-1 for idx in indices if -D8C_OFFSET<=idx<0)
    for idx,target,noise,path,_ in canonical.d8b_labels(d8b,directory=D8B):
        stat=path.stat()
        value.update(str(-idx-1).encode() + b'\0' + target[-1].tobytes() + np.float64(noise).tobytes())
        source=dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,head_tail_sha256=_head_tail_sha256(path))
        if source not in sources: sources.append(source)
    d8c=sorted(-idx-D8C_OFFSET-1 for idx in indices if idx<=-D8C_OFFSET-1)
    for idx,target,noise,path,_ in canonical.d8c_labels(d8c,directory=D8C):
        stat=path.stat()
        value.update(str(d8c_identity(idx)).encode()+b'\0'+target[-1].tobytes()+np.float64(noise).tobytes())
        source=dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,head_tail_sha256=_head_tail_sha256(path))
        if source not in sources: sources.append(source)
    identity=dict(schema='whest-label-corpus-v2',indices=indices,labels_sha256=value.hexdigest(),sources=sources)
    verify_corpus_identity(identity)
    return identity


def _file_version(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=4096)
def _versioned_head_tail_sha256(path, version, chunk):
    """Cache bytes only for a captured inode/size/mtime/ctime; never a pathname alone."""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        h.update(stream.read(chunk))
        size = stream.seek(0, 2)
        if size > chunk:
            stream.seek(max(chunk, size - chunk)); h.update(stream.read(chunk))
    h.update(str(size).encode())
    if _file_version(path) != version:
        raise ValueError('Training label corpus changed during fingerprint read: ' + path)
    return h.hexdigest()


def _head_tail_sha256(path, chunk=1 << 20):
    """Same fingerprint bytes; stable file versions require no repeated content I/O."""
    path = str(Path(path).resolve())
    version = _file_version(path)
    result = _versioned_head_tail_sha256(path, version, chunk)
    if _file_version(path) != version:
        raise ValueError('Training label corpus changed during fingerprint check: ' + path)
    return result


def verify_corpus_identity(identity):
    for source in identity.get('sources',()):
        try:
            stat=Path(source['path']).stat()
            current=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
            if 'head_tail_sha256' in source:
                current['head_tail_sha256']=_head_tail_sha256(source['path'])
        except OSError as exc:
            raise ValueError('Training label corpus source is unavailable') from exc
        if any(source[key]!=value for key,value in current.items()):
            raise ValueError('Training label corpus changed after its identity was frozen: '+source['path'])


def _full():
    return canonical.full_dataset()


def d8b_weights(i):
    g = torch.Generator(device='cpu').manual_seed(770000 + int(i))
    return torch.randn(DEPTH, WIDTH, WIDTH, generator=g, dtype=torch.float32) * (2.0 / WIDTH) ** 0.5


def _d8b_row(i):
    target,noise,_,_=canonical.d8b_label(int(i),directory=D8B)
    return target[-1],noise


def net(idx):
    """Return the canonical validated weights, final label and label-noise term."""
    if idx<=-D8C_OFFSET-1:
        index=-idx-D8C_OFFSET-1
        target,noise,_,_=canonical.d8c_label(index,directory=D8C)
        return d8c_weights(index),torch.as_tensor(target[-1]),noise
    namespace,index=('d8b',-idx-1) if idx<0 else ('full',idx)
    if namespace=='d8b':
        y,noise=_d8b_row(index)
        return d8b_weights(index),torch.as_tensor(y),noise
    weights,target,metadata=canonical.row(namespace,index)
    return torch.as_tensor(weights),torch.as_tensor(target[-1]),metadata['noise_mse']


def net_record(idx):
    """The evaluated tensors and their exact corpus seed/hash provenance."""
    namespace,index=('d8b',-idx-1) if idx<0 else ('full',idx)
    if idx<=-D8C_OFFSET-1: namespace,index='d8c',-idx-D8C_OFFSET-1
    weights,target,metadata=canonical.row(namespace,index)
    return torch.as_tensor(weights),torch.as_tensor(target[-1]),metadata['noise_mse'],metadata


def ranges(spec, d8b=False, d8c=False):
    """'0:800,832:840' or '800,809' → indices (d8b=True: negative d8b ids)."""
    out=[]
    for part in spec.split(','):
        if not part.strip(): continue
        values=[int(x) for x in part.split(':')]
        if len(values)==1: out.append(values[0])
        elif len(values) in (2,3): out.extend(range(*values))
        else: raise ValueError('Invalid network range: '+part)
    if d8c:
        if d8b: raise ValueError('Select exactly one corpus namespace')
        return [d8c_identity(i) for i in out]
    if d8b:
        # Out-of-range explicit d8b input must not become a valid identity in
        # another corpus (negative indices alias full; large indices alias d8c).
        if any(not 0 <= i < 14048 for i in out):
            raise ValueError('d8b index is outside its dataset')
        return [-(i + 1) for i in out]
    return out


def panel(spec):
    """Evaluation panel spec: full1000 ranges plus 'd8b:'-prefixed d8b ranges ('832:864', 'd8b:12536:13048', or both).
    A spec without 'd8b:' tokens is exactly ranges(spec)."""
    full, d8b = [], []
    for part in spec.split(','):
        part = part.strip()
        if part: (d8b if part.startswith('d8b:') else full).append(part.removeprefix('d8b:'))
    return ranges(','.join(full)) + ranges(','.join(d8b), d8b=True)


def labels_all(idx):
    """All-layer N = 1e9 means (depth, width) float64 for full1000 nets; None for d8b (final layer only)."""
    if idx < 0:
        return None
    return torch.as_tensor(np.asarray(_full().labels(int(idx))['all_layer_means'], dtype=np.float64))
