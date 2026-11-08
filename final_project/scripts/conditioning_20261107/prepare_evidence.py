"""Freeze prior-known orbit identities and reuse only certified exact scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.pool_scoring import cache_context, certify_cache, atomic_json
from cathode_cvae.mlip.registry import MODELS
from cathode_cvae.runtime import require_compute_allocation


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);a=p.parse_args();require_compute_allocation()
    out=a.root/'runs/conditioning_20261107';parent=a.root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    parent_sha=sha256(parent);previous=a.previous/'runs/overnight_20261107'
    data_dir=out/'data';data_dir.mkdir(exist_ok=True)
    with np.load(a.root/'artifacts/current/improvement_v1/training_round0.npz') as z:
        keys=z['canonical_masks'].astype(str);roles=z['split'].astype(str)
    train=sorted(set(keys[roles=='train']));history=set(keys)
    inputs=[]
    history_file=a.previous/'artifacts/overnight_20261107/data/history_assessment.json'
    if not history_file.exists():history_file=previous/'data/history_assessment.json'
    h=json.loads(history_file.read_text());history.update(h);inputs.append(dict(path=str(history_file),sha256=sha256(history_file),rows=len(h)))
    for name in ['reference','benchmark','demo']:
        path=(previous/'data/reference.npz' if name=='reference' else previous/f'{name}/candidates.npz')
        with np.load(path) as z:
            masks=z['masks'] if 'masks' in z else 1-z['occupancy']
        kk=[''.join(map(str,r.astype(int))) for r in masks]
        history.update(kk);inputs.append(dict(path=str(path),sha256=sha256(path),rows=len(kk)))
    for name,value in [('training_keys.json',train),('history_before_development.json',sorted(history))]:
        path=data_dir/name
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError('Preserving different existing history/data snapshot')
        if not path.exists():atomic_json(path,value)
    oracle=out/'oracle';oracle.mkdir(exist_ok=True)
    old_registry=json.loads((previous/'oracle/registry.json').read_text())
    if old_registry['parent_sha256']!=parent_sha:raise ValueError('Prior oracle has different physical parent')
    collected=previous/'collected_final';oldreceipt=json.loads((collected/'receipt.json').read_text())
    if oldreceipt['status']!='PASS' or oldreceipt['registry_sha256']!=sha256(previous/'oracle/registry.json'):
        raise ValueError('Prior score collection lacks a matching complete registry receipt')
    cache_receipts={}
    for model in ['uma','mace']:
        worker_files=[]
        for f in sorted((previous/'oracle/workers').glob('*.json')):
            w=json.loads(f.read_text())
            if w.get('model')!=model:continue
            spec=MODELS[model]
            if any([w.get('head')!=spec['head'],w.get('package_version')!=spec['version'],
                    w.get('dtype')!='float32',w.get('tf32') is not False,
                    w.get('checkpoint_hashes')!={Path(k).name:v for k,v in spec['files'].items()}]):
                raise ValueError('Prior worker teacher/head/precision does not match new physical cache')
            worker_files.append(dict(path=str(f),sha256=sha256(f)))
        if not worker_files:raise ValueError('No verified prior worker for teacher')
        prior_cache=previous/f'oracle/cache_{model}.npz'
        paths=[collected/f'{model}.npz']
        if prior_cache.exists():
            certify_cache(previous/'oracle',model,parent_sha);paths.insert(0,prior_cache)
        records={};ambiguous=set();sources=[]
        for path in paths:
            with np.load(path) as z:
                arrays={k:z[k] for k in ['structure_ids','energy_eV','n_atoms','n_Li']}
            for i,sid in enumerate(arrays['structure_ids'].astype(str)):
                row=tuple(float(arrays[k][i]) for k in ['energy_eV','n_atoms','n_Li'])
                if not np.isfinite(row).all():continue
                if sid in records and records[sid]!=row:ambiguous.add(sid)
                records[sid]=row
            sources.append(dict(path=str(path),sha256=sha256(path)))
        for sid in ambiguous:records.pop(sid,None)
        ids=sorted(records);vals=np.asarray([records[s] for s in ids])
        cache=oracle/f'cache_{model}.npz'
        np.savez_compressed(cache,structure_ids=np.asarray(ids),energy_eV=vals[:,0],
            n_atoms=vals[:,1].astype(int),n_Li=vals[:,2].astype(int))
        certificate=dict(context=cache_context(model,parent_sha),npz_sha256=sha256(cache),
            sources=sources,verified_worker_receipts=worker_files,
            previous_certificate_sha256=sha256(previous/f'oracle/cache_{model}.json') if prior_cache.exists() else None,
            previous_registry_sha256=sha256(previous/'oracle/registry.json'),
            dropped_ambiguous_exact_values=sorted(ambiguous),
            scope='Exact certified canonical geometries; scalar energies only, no force rotation/alias')
        atomic_json(oracle/f'cache_{model}.json',certificate)
        cache_receipts[model]=dict(cached=len(ids),ambiguous=len(ambiguous),sha256=sha256(cache))
    receipt=dict(status='PASS',training_keys=len(train),known_before_development=len(history),
        prior_sources=inputs,caches=cache_receipts,
        protected_roles='Original orbit roles unchanged; no test labels used for selection',
        history_policy='Prior examined assessments and new development are known history for final novelty',
        data_outputs={p.name:sha256(p) for p in data_dir.iterdir() if p.is_file() and p.name!='receipt.json'})
    atomic_json(data_dir/'receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
