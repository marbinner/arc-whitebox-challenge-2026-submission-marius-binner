"""Read cached full labels or d8b labels; no weights or labels go into code."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

import numpy as np
if __package__:
    from .common import SCRATCH, bulk_dir, digest, exclusive_lock, write_json
else:
    from common import SCRATCH, bulk_dir, digest, exclusive_lock, write_json

D8B = Path(os.environ.get('WHEST_D8B', str(Path(os.environ.get('WHEST_DATA', 'data'))/'d8b')))


@lru_cache(maxsize=1)
def full_dataset():
    """A lazy index over the prepared split; fresh workers open only their shard."""
    return PreparedRows()


def prepared_directory():
    override = os.environ.get('WHEST_FULL_DATASET')
    if override:
        return Path(override).expanduser().resolve()
    local = Path(os.environ.get('WHEST_FULL_LOCAL', 'data/hf-full-v2-phase2/full'))
    if local.is_dir():
        return local
    hub = Path(os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or
               str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub'))
    repository = hub/'datasets--aicrowd--arc-whestbench-public-2026'
    revision = (repository/'refs/v2-phase2').read_text().strip()
    root = repository/'snapshots'/revision
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['n_samples'] != 10**9 or metadata['width'] != 1024 or metadata['depth'] != 16:
        raise ValueError('Unexpected full dataset metadata')
    return root/metadata['prepared_splits']['full']['path']


class PreparedRows:
    def __init__(self):
        self.directory = prepared_directory()
        self._lock = threading.RLock()
        self._fingerprint = None
        self._refresh()

    def _source(self):
        state = (self.directory/'state.json').read_bytes()
        shards = []
        for item in json.loads(state)['_data_files']:
            stat = (self.directory/item['filename']).stat()
            shards.append(dict(filename=item['filename'], device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
        return dict(schema='wb-prepared-index-v2',directory=str(self.directory.resolve()),
                    state_sha256=hashlib.sha256(state).hexdigest(),shards=shards)

    @staticmethod
    def _valid(value, source):
        if not isinstance(value,dict) or value.get('source') != source:
            return False
        start = 0
        shards = value.get('shards')
        if not isinstance(shards,list) or len(shards) != len(source['shards']):
            return False
        for shard,original in zip(shards,source['shards']):
            if (not isinstance(shard,dict) or shard.get('filename') != original['filename']
                    or type(shard.get('start')) is not int or shard['start'] != start
                    or type(shard.get('stop')) is not int or shard['stop'] < start):
                return False
            start = shard['stop']
        return type(value.get('rows')) is int and value['rows'] == start

    def _refresh(self):
        from datasets import Dataset
        with self._lock:
            source = self._source()
            key = hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
            if key == self._fingerprint:
                return
            index = bulk_dir(SCRATCH/'dataset-index')/(key+'.json')
            with exclusive_lock(index.with_suffix('.lock')):
                try:
                    value = json.loads(index.read_text())
                except (OSError,ValueError):
                    value = None
                if not self._valid(value,source):
                    shards, start = [], 0
                    for item in source['shards']:
                        dataset = Dataset.from_file(str(self.directory/item['filename']))
                        shards.append(dict(filename=item['filename'],start=start,stop=start+len(dataset)))
                        start += len(dataset)
                    if self._source() != source:
                        raise ValueError('Prepared shards changed during indexing')
                    value = dict(rows=start,shards=shards,source=source)
                    write_json(index,value)
            self.index = value
            self._fingerprint = key

    def __len__(self):
        self._refresh()
        return self.index['rows']

    def __getitem__(self, index):
        from datasets import Dataset
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = next(s for s in self.index['shards'] if s['start'] <= index < s['stop'])
        dataset = Dataset.from_file(str(self.directory/shard['filename'])).with_format('numpy')
        if len(dataset) != shard['stop']-shard['start']:
            raise ValueError('Prepared shard changed since indexing')
        return dataset[index-shard['start']]

    def labels(self, index):
        from datasets import Dataset
        if type(index) is not int or index < 0 or index >= len(self): raise IndexError(index)
        shard = next(s for s in self.index['shards'] if s['start'] <= index < s['stop'])
        dataset = Dataset.from_file(str(self.directory/shard['filename']))
        dataset = dataset.select_columns(['all_layer_means','mlp_seed','avg_variance']).with_format('numpy')
        if len(dataset) != shard['stop']-shard['start']: raise ValueError('Prepared label shard changed')
        return dataset[index-shard['start']]


def d8b_signature(directory=None):
    signature=[]
    for path in sorted((Path(directory) if directory else D8B).glob('shard*.npz')):
        stat=path.stat()
        signature.append((str(path.resolve()),stat.st_dev,stat.st_ino,stat.st_size,
                          stat.st_mtime_ns,stat.st_ctime_ns))
    return tuple(signature)


def d8b_index(directory=None):
    return _d8b_index(d8b_signature(directory))


@lru_cache(maxsize=1)
def _d8b_index(signature):
    index = {}
    for filename,*expected in signature:
        path=Path(filename)
        with np.load(path, allow_pickle=False) as z:
            ids = z['idx']
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= 14048).any():
                raise ValueError('d8b idx must contain bounded integer identities')
            for name in ('w_seed0','n_samples','n_done'):
                if z[name].shape != () or z[name].dtype.kind not in 'iu':
                    raise ValueError('d8b metadata must be scalar integers: '+name)
            if int(z['w_seed0']) != 770000 or int(z['n_samples']) != 10**8 or int(z['n_done']) != len(ids):
                raise ValueError('Unexpected d8b regeneration/label metadata: ' + str(path))
            means,moments=z['final_mean'],z['moments']
            if (means.shape!=(len(ids),1024) or moments.ndim!=4 or moments.shape[0]!=len(ids)
                    or moments.shape[1]<2 or moments.shape[2:]!=(16,1024)
                    or means.dtype.kind not in 'fiu' or moments.dtype.kind not in 'fiu'):
                raise ValueError('Invalid d8b label dimensions/dtypes: '+str(path))
            variance=moments[:,1,-1].astype(np.float64)-moments[:,0,-1].astype(np.float64)**2
            if not np.isfinite(means).all() or not np.isfinite(moments).all() or (variance < -1e-7).any():
                raise ValueError('Invalid d8b labels/variance: '+str(path))
            for j, idx in enumerate(ids):
                if int(idx) in index:
                    raise ValueError('Duplicate d8b index: ' + str(int(idx)))
                index[int(idx)] = (path, j)
        stat=path.stat()
        if list((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)) != expected:
            raise ValueError('d8b shard changed during indexing: '+str(path))
    return index


def d8b_labels(indices,directory=None):
    """Read selected labels once per shard, binding offsets to captured versions."""
    identities=sorted(set(indices))
    if not identities: return
    signature=d8b_signature(directory);index=_d8b_index(signature)
    grouped={}
    for identity in identities:
        if type(identity) is not int or identity<0: raise ValueError('d8b row requires an integer identity')
        path,offset=index[identity]
        grouped.setdefault(path,[]).append((identity,offset))
    for path,rows in grouped.items():
        with np.load(path,allow_pickle=False) as z:
            ids,moments,means=z['idx'],z['moments'],z['final_mean']
        if d8b_signature(directory)!=signature: raise ValueError('d8b shard changed after index capture')
        for identity,offset in rows:
            if int(ids[offset])!=identity: raise ValueError('d8b row identity changed since indexing')
            m=moments[offset].astype(np.float64)
            target=m[0].copy();target[-1]=means[offset]
            variance=m[1,-1]-m[0,-1]**2
            if (target.shape!=(16,1024) or not np.isfinite(target).all()
                    or not np.isfinite(variance).all() or (variance < -1e-7).any()):
                raise ValueError('Invalid d8b row label/variance')
            yield identity,target,float(np.maximum(variance,0).mean())/10**8,path,offset
    if d8b_signature(directory)!=signature: raise ValueError('d8b label corpus changed during read')


def d8b_label(index,directory=None):
    _,target,noise,path,offset=list(d8b_labels([index],directory))[0]
    return target,noise,path,offset


D8C = Path(os.environ.get('WHEST_D8C', str(Path(os.environ.get('WHEST_DATA', 'data'))/'d8c')))
D8C_INDEX_BOUND = 1_000_000
D8C_W_SEED0, D8C_S_SEED0, D8C_SAMPLES = 880000, 9900000, 10**7
D8C_PRECISION_MODES = ('fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-b6291456-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-aggshift-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-aggshift-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-persistphase-w4-b1048576-splitgauss2-gain4-ok-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-persistphase-w4-netgraph-b8388608-splitgauss2-gain4-ok-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-phaseasync-w16-phasem256-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-w16-phasem256-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced16-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced256-cachephase-b2097152-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced4-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced4-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-nativeepi-epic64-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced4-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-nativeepi-epic64-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced4-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-persistphase-rp4-inputwide-w4-b1048576-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-persistphase-rp4-inputwide-w4-netgraph-b8388608-splitgauss2-gain4-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-phasepipeline-w8-b1048576-splitgauss2-gain4-ok-asyncwrite-prefetch', 'fp16sr-cublasphase-stoch-balanced8-cachephase-meancorrect-carrymean-layeragg-i64-streamout-striderows-directcache-aggmean-coopshift-firstnorm-wmmaphase-phasepipeline-w8-netgraph-b8388608-splitgauss2-gain4-ok-asyncwrite-prefetch', 'fp16sr-packed-fused-tall-k256-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss2-aggregate-b6291456-sharephase-shareblocks-cachephase-bitcache-bitreverse-rowgroup4-gain3-fdirect-chunkagg-asyncwrite-prefetch', 'fp16sr-packed-fused-tall-k256-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss2-streammom-aggregate-b4194304-sharephase-shareblocks-cachephase-bitcache-bitreverse-rowgroup4-chunkagg-asyncwrite-prefetch', 'fp16sr-packed-fused-tall-stoch-balanced256-precomp-circular-repeat-aggregate-b131072-prefetch', 'fp16sr-packed-fused-tall-stoch-balanced4-aggregate-prefetch', 'fp16sr-packed-fused-tall-stoch-balanced4-precomp-aggregate-prefetch', 'fp16sr-packed-fused-tall-stoch-balanced8-precomp-aggregate-b131072-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-directgauss-streammom-aggregate-b131072-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-directgauss-streammom-aggregate-b524288-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-directgauss-streammom-aggregate-b524288-sharephase-asyncwrite-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-directgauss-streammom-aggregate-b524288-sharephase-cachephase-asyncwrite-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss-streammom-aggregate-b1048576-sharephase-cachephase-bitcache-asyncwrite-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-expand-sort-streammom-aggregate-b131072-prefetch', 'fp16sr-packed-fused-wide128-stoch-balanced256-precomp-circular-repeat-streammom-aggregate-b131072-prefetch', 'fp16sr-packed-fused-wide128-w8-k128-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss2-streammom-aggregate-b2097152-sharephase-shareblocks-cachephase-bitcache-asyncwrite-prefetch', 'fp16sr-packed-fused-wide128-w8-k128-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss2-streammom-aggregate-b4194304-sharephase-shareblocks-cachephase-bitcache-bitreverse-asyncwrite-prefetch', 'fp16sr-packed-fused-wide128-w8-k256-stoch-balanced256-precomp-circular-repeat-expand-sort-splitgauss2-streammom-aggregate-b4194304-sharephase-shareblocks-cachephase-bitcache-bitreverse-chunkagg-asyncwrite-prefetch', 'fp16w2-packed-fused-tall-aggregate-prefetch', 'fp16w2-packed-fused-tall-k64', 'fp16w2-packed-fused-w16-k64', 'fp16w2-packed-k64', 'fp16w2-packed-small', 'fp16w2-packed-w16-k64', 'fp16w2-rnacc', 'fp16w2-rnacc-k64', 'fp16x3-rnacc', 'ieee', 'tf32x3', 'tf32x3-local', 'tf32x3-wide')


def d8c_signature(directory=None):
    signature=[]
    for path in sorted((Path(directory) if directory else D8C).glob('shard*.npz')):
        stat=path.stat()
        signature.append((str(path.resolve()),stat.st_dev,stat.st_ino,stat.st_size,
                          stat.st_mtime_ns,stat.st_ctime_ns))
    return tuple(signature)


def d8c_index(directory=None):
    return _d8c_index(d8c_signature(directory))


@lru_cache(maxsize=1)
def _d8c_index(signature):
    index={}
    for filename,*expected in signature:
        path=Path(filename)
        with np.load(path,allow_pickle=False) as z:
            required={'idx','w_seed0','s_seed0','n_samples','n_done','moments','final_mean',
                      'noise_rep_final','sample_seeds','schema','precision'}
            if not required<=set(z.files): raise ValueError('Incomplete d8c shard: '+str(path))
            ids=z['idx']
            if (ids.ndim!=1 or ids.dtype!=np.dtype('int64') or not len(ids)
                    or (ids<0).any() or (ids>=D8C_INDEX_BOUND).any()):
                raise ValueError('d8c idx must contain bounded int64 identities')
            for name in ('w_seed0','s_seed0','n_samples','n_done'):
                if z[name].shape!=() or z[name].dtype.kind not in 'iu':
                    raise ValueError('d8c metadata must be scalar integers: '+name)
            if (int(z['w_seed0'])!=D8C_W_SEED0 or int(z['s_seed0'])!=D8C_S_SEED0
                    or int(z['n_samples'])!=D8C_SAMPLES or int(z['n_done'])!=len(ids)):
                raise ValueError('Unexpected d8c regeneration/label metadata: '+str(path))
            if z['schema'].shape!=() or str(z['schema'])!='whest-d8c-v1':
                raise ValueError('Invalid d8c schema')
            if z['precision'].shape!=() or str(z['precision']) not in D8C_PRECISION_MODES:
                raise ValueError('Unqualified d8c precision')
            seeds=z['sample_seeds']
            if seeds.dtype!=np.dtype('int64') or seeds.shape!=ids.shape or not np.array_equal(seeds,D8C_S_SEED0+ids):
                raise ValueError('Invalid d8c sample seed identities')
            means,moments,repeat=z['final_mean'],z['moments'],z['noise_rep_final']
            if (means.shape!=(len(ids),1024) or moments.shape!=(len(ids),4,16,1024)
                    or means.dtype!=np.dtype('float64') or moments.dtype!=np.dtype('float32')
                    or repeat.shape not in ((0,),(1024,)) or repeat.dtype!=np.dtype('float64')):
                raise ValueError('Invalid d8c label dimensions/dtypes: '+str(path))
            variance=moments[:,1,-1].astype(np.float64)-moments[:,0,-1].astype(np.float64)**2
            if (not np.isfinite(means).all() or not np.isfinite(moments).all()
                    or not np.isfinite(repeat).all() or (variance < -1e-7).any()
                    or (moments < 0).any() or (means < 0).any()):
                raise ValueError('Invalid d8c labels/variance: '+str(path))
            for j,identity in enumerate(ids):
                if int(identity) in index: raise ValueError('Duplicate d8c index: '+str(int(identity)))
                index[int(identity)]=(path,j)
        stat=path.stat()
        if list((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))!=expected:
            raise ValueError('d8c shard changed during indexing: '+str(path))
    return index


def d8c_labels(indices,directory=None):
    """Selected d8c labels, exact final means and Var(h16)/1e7 noise terms."""
    identities=sorted(set(indices))
    if not identities: return
    if any(type(i) is not int or not 0<=i<D8C_INDEX_BOUND for i in identities):
        raise ValueError('d8c row requires a bounded integer identity')
    signature=d8c_signature(directory);index=_d8c_index(signature)
    grouped={}
    for identity in identities:
        path,offset=index[identity]
        grouped.setdefault(path,[]).append((identity,offset))
    for path,rows in grouped.items():
        with np.load(path,allow_pickle=False) as z:
            ids,moments,means=z['idx'],z['moments'],z['final_mean']
        if d8c_signature(directory)!=signature: raise ValueError('d8c shard changed after index capture')
        for identity,offset in rows:
            if int(ids[offset])!=identity: raise ValueError('d8c row identity changed since indexing')
            m=moments[offset].astype(np.float64)
            target=m[0].copy();target[-1]=means[offset]
            variance=m[1,-1]-m[0,-1]**2
            yield identity,target,float(np.maximum(variance,0).mean())/D8C_SAMPLES,path,offset
    if d8c_signature(directory)!=signature: raise ValueError('d8c label corpus changed during read')


def d8c_label(index,directory=None):
    _,target,noise,path,offset=list(d8c_labels([index],directory))[0]
    return target,noise,path,offset



def row(dataset, index):
    if dataset == 'full':
        value = full_dataset()[index]
        weights = np.asarray(value['weights'], dtype=np.float32)
        target = np.asarray(value['all_layer_means'], dtype=np.float64)
        noise = float(value['avg_variance']) / 10**9
        seed = int(value['mlp_seed'])
        metadata = dict(dataset='aicrowd/arc-whestbench-public-2026@v2-phase2/full',
                        noise_source='avg_variance / 1e9', label_samples=10**9)
    elif dataset == 'd8b':
        import torch
        torch.set_num_threads(1)
        target,noise,path,j = d8b_label(index)
        generator = torch.Generator(device='cpu').manual_seed(770000 + index)
        weights = (torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
                   * (2.0 / 1024)**0.5).numpy()
        seed = 770000 + index
        metadata = dict(dataset='keenanpepper/arc-whestbench-p2-d8b-corpus-14k',
                        shard=str(path), shard_row=j, label_samples=10**8,
                        noise_source='mean(final m2 - final m1 squared) / 1e8',
                        regeneration='torch CPU float32 randn(16,1024,1024), seed 770000+idx, scale sqrt(2/1024)')
    elif dataset == 'd8c':
        import torch
        target,noise,path,j=d8c_label(index)
        generator=torch.Generator(device='cpu').manual_seed(D8C_W_SEED0+index)
        weights=(torch.randn(16,1024,1024,generator=generator,dtype=torch.float32)
                 *(2.0/1024)**0.5).numpy()
        seed=D8C_W_SEED0+index
        metadata=dict(dataset='whest-d8c-v1',shard=str(path),shard_row=j,
                      label_samples=D8C_SAMPLES,
                      noise_source='mean(final m2 - final m1 squared) / 1e7',
                      regeneration='torch CPU float32 randn(16,1024,1024), seed 880000+idx, scale sqrt(2/1024)')
    else:
        raise ValueError('Unknown dataset: ' + dataset)
    if weights.shape != (16, 1024, 1024) or target.shape != (16, 1024):
        raise ValueError('Unexpected production data shape')
    if not (np.isfinite(weights).all() and np.isfinite(target).all() and np.isfinite(noise) and noise >= 0):
        raise ValueError('Non-finite production data')
    metadata.update(dataset_key=dataset, net=index, seed=seed, noise_mse=noise,
                    weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                    target_sha256=hashlib.sha256(target.tobytes()).hexdigest())
    return weights, target, metadata


def fixture_signature(dataset, index):
    """Prepared revisions are immutable; include file identity to detect local replacements."""
    source = dict(dataset=dataset, index=index, loader_sha256=digest(Path(__file__)))
    if dataset == 'full':
        rows = full_dataset()
        if index < 0 or index >= len(rows):
            raise IndexError(index)
        shard = next(s for s in rows.index['shards'] if s['start'] <= index < s['stop'])
        path = rows.directory/shard['filename']
        source.update(directory=str(rows.directory), state_sha256=digest(rows.directory/'state.json'))
        source['index_fingerprint'] = rows._fingerprint
    elif dataset == 'd8b':
        from importlib.metadata import version
        path, offset = d8b_index()[index]
        source.update(shard_row=offset, torch_version=version('torch'))
    elif dataset == 'd8c':
        from importlib.metadata import version
        path,offset=d8c_index()[index]
        source.update(shard_row=offset,torch_version=version('torch'))
    else:
        raise ValueError('Unknown dataset: '+dataset)
    stat = path.stat()
    source.update(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  inode=stat.st_ino, ctime_ns=stat.st_ctime_ns)
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def fixture_metadata(path):
    with np.load(path, allow_pickle=False) as value:
        return json.loads(str(value['metadata']))


def export_fixture(array,out,expected_sha256):
    """Preserve occupied destinations and remove incomplete newly created copies."""
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    created=False
    try:
        try:
            os.link(array,out);created=True
        except FileExistsError:
            pass
        except OSError:
            try:
                with out.open('xb') as target:
                    created=True
                    with Path(array).open('rb') as source:
                        shutil.copyfileobj(source,target)
                out.chmod(0o444)
            except FileExistsError:
                pass
        if digest(out)!=expected_sha256:
            raise ValueError('Fixture destination contains different bytes: '+str(out))
    except BaseException:
        if created:out.unlink(missing_ok=True)
        raise


def fixture(dataset, index, out=None):
    """Share an immutable, integrity-checked fixture across fresh client processes."""
    out = Path(out) if out is not None else None
    signature=fixture_signature(dataset,index)
    cache = bulk_dir(SCRATCH/'fixtures')/signature
    def verify_source():
        if fixture_signature(dataset,index)!=signature:
            raise ValueError('Prepared data source changed during fixture preparation')
    array, receipt = cache.with_suffix('.npz'), cache.with_suffix('.json')
    with exclusive_lock(cache.with_suffix('.lock')):
        meta = None
        try:
            value = json.loads(receipt.read_text())
            if value['fixture_sha256'] == digest(array) and value['metadata'] == fixture_metadata(array):
                meta = value['metadata']
                expected_sha256=value['fixture_sha256']
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if meta is None:
            weights, target, meta = row(dataset, index)
            verify_source()
            temporary = array.with_name(array.name+'.tmp-'+str(os.getpid())+'.npz')
            try:
                np.savez(temporary, weights=weights, target=target, seed=np.int64(meta['seed']),
                         metadata=np.asarray(json.dumps(meta, sort_keys=True)))
                verify_source()
                temporary.chmod(0o444)
                temporary.replace(array)
            finally:
                temporary.unlink(missing_ok=True)
            expected_sha256=digest(array)
            write_json(receipt, dict(metadata=meta, fixture_sha256=expected_sha256))
        verify_source()
        if out is None:
            return meta
        export_fixture(array,out,expected_sha256)
    return meta



