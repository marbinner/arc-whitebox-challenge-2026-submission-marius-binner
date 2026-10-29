"""Finite C-order API assets, PARAMS_DIFF, and optional runtime configuration."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .runtime import literal,merge_configuration
from ..export_manifest import verify_export
from .data import normalize_provenance


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def export_state(twin,state,out,*,checkpoint=None,config=None):
    out=Path(out)
    if out.is_symlink() or out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Export directory must be empty: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=out.name+'-export-',dir=out.parent))
    try:
        receipt=_write_export(twin,state,staging,checkpoint=checkpoint,config=config)
        verify_export(staging)
        # Replacing an empty directory is atomic; a concurrent populated target
        # makes os.replace fail without removing its contents.
        os.replace(staging,out)
        return receipt
    finally:
        if staging.exists(): shutil.rmtree(staging)


def _write_export(twin,state,out,*,checkpoint=None,config=None):
    with torch.no_grad():
        request=merge_configuration(state.request(evaluation=True),config)
        twin.model.update_runtime(**request)
        groups=twin.runtime_assets()
        schema=twin.model.runtime_schema()
        arrays={}
        for group,members in groups.items():
            arrays[group]={}
            for key,value in members.items():
                spec=schema['assets'][group+'.'+key]
                array=np.array(value.detach().cpu().numpy(),dtype=spec['dtype'],order='C',copy=True)
                if tuple(array.shape)!=tuple(spec['shape']) or not np.isfinite(array).all():
                    raise ValueError('Invalid export tensor: '+group+'.'+key)
                arrays[group][key]=array
        params={key:literal(value) for key,value in request['params'].items() if key in state.params}
        baseline=getattr(state,'source_params',state.defaults)
        changed={key:{'old':literal(baseline[key]),'new':value} for key,value in params.items()
                 if (twin.normalize_parameter(key,value)!=twin.normalize_parameter(key,baseline[key])
                     or twin.normalize_parameter(key,value)!=twin.normalize_parameter(key,state.defaults[key]))}
        for group,members in arrays.items():
            filenames={schema['assets'][group+'.'+key].get('asset') or group+'.npz' for key in members}
            if len(filenames)!=1:
                raise ValueError('Asset group spans multiple filenames: '+group)
            filename=next(iter(filenames))
            if filename!=group+'.npz':
                raise ValueError('Asset filename must identify its runtime group: '+group)
            np.savez(out/filename,**members)
        write_json(out/'PARAMS_DIFF.json',changed)
        write_json(out/'TRAINING-PROVENANCE.json',normalize_provenance(getattr(state,'training_provenance',None)))
        configuration=merge_configuration(twin.config,config)
        # Continuous literals live in PARAMS_DIFF; retain configured discrete
        # choices, which are not optimizer leaves, in the runtime request.
        configured_params=configuration.pop('params',{})
        choices={key:value for key,value in configured_params.items() if key not in params}
        if choices: configuration['params']=choices
        if request.get('corrector'):
            configuration['corrector']={**configuration.get('corrector',{}),**request['corrector']}
        if request.get('per_layer'): configuration['per_layer']=request['per_layer']
        if configuration or config or twin.config:
            write_json(out/'RUNTIME_CONFIG.json',literal(configuration))
    files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir()) if path.is_file()}
    hashes={name:sha for name,sha in files.items() if name.endswith('.npz')}
    receipt=dict(schema='whest-train-export-v2',complete=True,files_sha256=files,
                 runtime_api=1,checkpoint=str(checkpoint) if checkpoint else None,
                 base=str(twin.base),source_sha256=twin.source_sha256,params_changed=sorted(changed),
                 npz_all_c_contiguous=True,arrays={group:{key:dict(shape=list(v.shape),dtype=str(v.dtype))
                    for key,v in members.items()} for group,members in arrays.items()},
                 assets_sha256=hashes,selected=state.selected)
    write_json(out/'EXPORT-RECEIPT.json',receipt)
    return receipt


def import_request(folder,normalize,*,legacy=False):
    """Read an export into a partial runtime request; original NPZ names retained."""
    folder=Path(folder)
    verify_export(folder,legacy=legacy)
    diff=json.loads((folder/'PARAMS_DIFF.json').read_text())
    if 'm2_params_keys' in diff:
        diff={**diff['m2_params_keys'],**diff.get('inlined_in_M1_until_restored',{})}
    params={key:normalize(key,value['new'] if isinstance(value,dict) and 'new' in value else value)
            for key,value in diff.items()}
    # Old fe34 diffs contain routing metadata which is a separate discrete
    # configuration in the clean base; its continuous value is inner_k4.
    if 'fe34' in params: params['fe34']={'inner_k4':params['fe34']['inner_k4']}
    assets={}
    for path in sorted(folder.glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            assets[path.stem]={key:np.array(z[key],order='C',copy=True) for key in z.files}
    request=dict(params=params,assets=assets)
    configuration=folder/'RUNTIME_CONFIG.json'
    if configuration.exists(): request=merge_configuration(request,json.loads(configuration.read_text()))
    return request
