"""Completion and byte integrity for atomically published trainer exports."""
import hashlib
import json
from pathlib import Path


def normalize_provenance(value):
    """Canonical namespaced ancestry; absence never proves held-out labels."""
    if value is None:value=dict(seen_indices=[],known=False,corpus_sha256=[])
    if not isinstance(value,dict):raise ValueError('Training provenance must be an object')
    if value.get('schema','whest-training-provenance-v1')!='whest-training-provenance-v1':
        raise ValueError('Unsupported training provenance schema')
    population=value.get('seen_indices',[])
    if (not isinstance(population,list) or any(type(index) is not int or not (-14048<=index<1000 or -2_000_000<=index<-1_000_000) for index in population)
            or type(value.get('known',False)) is not bool):
        raise ValueError('Invalid training provenance identities or known flag')
    corpora=value.get('corpus_sha256',[])
    if not isinstance(corpora,list) or any(not isinstance(item,str) or not item for item in corpora):
        raise ValueError('Invalid training provenance corpus identities')
    return dict(schema='whest-training-provenance-v1',seen_indices=sorted(set(population)),
                known=value.get('known',False),corpus_sha256=sorted(set(corpora)))


def merge_provenance(*values):
    values=[normalize_provenance(value) for value in values]
    return normalize_provenance(dict(seen_indices=sorted({idx for value in values for idx in value['seen_indices']}),
        known=all(value['known'] for value in values),
        corpus_sha256=sorted({sha for value in values for sha in value['corpus_sha256']})))


def package_provenance(package):
    path=Path(package)/'TRAINING-PROVENANCE.json'
    return normalize_provenance(json.loads(path.read_text()) if path.exists() else None)


def verify_export(folder, *, legacy=False):
    folder=Path(folder)
    receipt=folder/'EXPORT-RECEIPT.json'
    try:
        value=json.loads(receipt.read_text())
    except (OSError,ValueError):
        if legacy: return None
        raise ValueError('Export completion manifest is missing; legacy exports require --legacy-export')
    if value.get('schema')!='whest-train-export-v2':
        if legacy: return None
        raise ValueError('Legacy export completion format requires --legacy-export')
    files=value.get('files_sha256')
    if value.get('complete') is not True or not isinstance(files,dict) or 'PARAMS_DIFF.json' not in files:
        raise ValueError('Invalid export completion manifest')
    inventory={path.name for path in folder.iterdir() if path.name!='EXPORT-RECEIPT.json'}
    if inventory!=set(files): raise ValueError('Export completion manifest inventory mismatch')
    for name,expected in files.items():
        if Path(name).name!=name or not (folder/name).is_file():
            raise ValueError('Invalid export manifest filename')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Export integrity mismatch: '+name)
    return value
