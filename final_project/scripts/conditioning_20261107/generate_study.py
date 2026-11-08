"""Prespecified prior proposals and paired condition/latent/posterior diagnostics.

Energy is never used to filter generated proposals. Main assessment attempts are
kept separate from diagnostic proposals, but every physical query is accounted.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from cathode_cvae.conditions import classify, load_schema
from cathode_cvae.data import sha256
from cathode_cvae.learning.generate_mask import load_model, resolve_request, attempt_records
from cathode_cvae.learning.train_mask import canonicalize
from cathode_cvae.runtime import require_compute_allocation


def seed_for(*items):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,items)).encode()).digest()[:4],'little')


def write_rows(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def validate_frozen_manifest(manifest, *, parent_sha, data_sha, schema_id, namespace,
                             count, diagnostics):
    """Refuse changes to the prespecified final parent, history and draw budget."""
    if manifest.get('status') != 'FROZEN_BEFORE_FINAL_GENERATION':
        return
    actual = dict(expected_parent_sha256=parent_sha,
                  expected_training_data_sha256=data_sha,
                  expected_condition_schema_id=schema_id, namespace=namespace,
                  count_per_composition_bin_replicate=count, diagnostics=diagnostics)
    for key, value in actual.items():
        if manifest.get(key) != value:
            raise ValueError(f'Frozen final manifest {key} differs from execution')
    history = manifest.get('history_path')
    if not history or sha256(history) != manifest.get('history_sha256'):
        raise ValueError('Frozen final assessment history changed before generation')
    expected = len(manifest['models'])*len(manifest['counts'])*len(manifest['requested_bins'])*count
    if expected != manifest.get('expected_primary_attempts'):
        raise ValueError('Frozen final attempt budget is inconsistent')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--namespace',required=True);p.add_argument('--count',type=int,default=64)
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--device',default='cuda')
    a=p.parse_args();require_compute_allocation()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    parent_path=a.root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    parent_sha=sha256(parent_path)
    with np.load(parent_path) as z:inv=torch.tensor(np.argsort(z['permutations'],axis=1),device=a.device)
    data_path=a.root/'artifacts/current/improvement_v1/training_round0.npz'
    data_sha=sha256(data_path)
    with np.load(data_path) as z:data={k:z[k] for k in z.files}
    schema=load_schema(a.root/'artifacts/overnight_20261107/data/condition_schema_uma.json')
    labels=classify(schema,data['n_Li'],data['energy_eV_atom'])
    manifest=json.loads(a.manifest.read_text())
    validate_frozen_manifest(manifest,parent_sha=parent_sha,data_sha=data_sha,
        schema_id=schema['identity'],namespace=a.namespace,count=a.count,diagnostics=a.diagnostics)
    for key,actual in [('parent_sha256',parent_sha),('data_sha256',data_sha),
                       ('namespace',a.namespace),('count_per_group',a.count)]:
        if key in manifest and manifest[key]!=actual:
            raise ValueError(f'Frozen manifest {key} differs from execution')
    if manifest.get('history_path') and sha256(manifest['history_path'])!=manifest['history_sha256']:
        raise ValueError('Frozen assessment history changed before generation')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    entries=manifest['models'];rows=[];raw=[];seed_raw=[];frozen=[]
    train=data['split']=='train';started=time.time()

    def append(x,entry,n,requested,diagnostic,random_seed,cksha,cfg,seeds=None,extra=None):
        x=np.asarray(x,dtype=np.uint8)
        if not np.isin(x,[0,1]).all() or not np.all(x.sum(1)==n):
            raise ValueError('Exact binary count violated; do not repair/reject silently')
        _,codes=canonicalize(torch.tensor(x,device=a.device,dtype=torch.float32),inv)
        keys=[format(int(c),'048b') for c in codes];ids=[f'mask48_{int(c):012x}' for c in codes]
        mode=cfg.get('condition_mode') or ('composition' if cfg.get('condition_dim')==1 else 'energy')
        effective=(requested if mode=='multibin' else
                   'unconditioned' if mode=='composition' else 'low' if requested=='low' else 'rest')
        rr=attempt_records(len(x),ids,keys,method=entry['method'],fit_seed=entry.get('fit_seed',''),
            sampling_seed=random_seed,n_target=n,requested_bin=requested,
            effective_model_class=effective if cfg else 'frozen-sampler',schema_id=schema['identity'],
            checkpoint_sha256=cksha,parent_sha256=parent_sha,
            geometry_protocol='pristine-derived deletion; unrelaxed; exact canonical parent coordinates')
        for i,r in enumerate(rr):
            j=len(raw);raw.append(x[i]);seed_raw.append(np.zeros(48,np.uint8) if seeds is None else seeds[i])
            r.update(attempt_id=f'{a.namespace}-{entry["method"]}-{entry.get("replicate_id",entry.get("fit_seed",0))}-{n}-{requested}-{diagnostic}-{i}',
                replicate_id=entry.get('replicate_id',entry.get('fit_seed',0)),diagnostic=diagnostic,
                raw_row=j,paired_id=f'{a.namespace}-{entry.get("replicate_id",entry.get("fit_seed",0))}-{n}-{i}',
                allocation_policy=a.namespace,allocation_run_id=f'{entry["method"]}-{n}',allocation_weight=1/3,
                descriptor_range=entry.get('descriptor_range',''))
            if extra:r.update(extra[i] if isinstance(extra,list) else extra)
            rows.append(r)

    for entry in entries:
        ck=Path(entry['checkpoint']) if entry.get('checkpoint') else None
        if ck:
            expected=entry.get('expected_checkpoint_sha256',entry.get('checkpoint_sha256'))
            if expected and sha256(ck)!=expected:
                raise ValueError('Checkpoint differs from frozen selection before generation')
            model,cfg=load_model(ck,a.device);cksha=sha256(ck)
            frozen.append(dict(entry,checkpoint_sha256=cksha))
        else:model=None;cfg={};cksha='not-applicable'
        for n in [12,24,36]:
            rep=entry.get('replicate_id',entry.get('fit_seed',0))
            common_seed=seed_for(a.namespace,rep,n,'common')
            g=torch.Generator(device=a.device).manual_seed(common_seed)
            z32=torch.randn((a.count,32),generator=g,device=a.device)
            u=torch.rand((a.count,48),generator=g,device=a.device)
            latent_dim=cfg.get('latent_dim',0)
            if latent_dim>32:raise ValueError('Protocol common latent draws cover at most32dimensions')
            for requested in ['low','middle','high']:
                if model is not None:
                    kwargs={'descriptor_range':entry['descriptor_range']} if entry.get('descriptor_range') else {}
                    vector,*_=resolve_request(cfg,n,requested,None,schema,**kwargs)
                    c=torch.tensor(vector,device=a.device).repeat(a.count,1)
                    with torch.no_grad():x=model.sample(c,n,z=z32[:,:latent_dim],uniforms=u).cpu().numpy()
                    append(x,entry,n,requested,'primary',common_seed,cksha,cfg,
                        extra=dict(latents_sha256=hashlib.sha256(z32.cpu().numpy().tobytes()).hexdigest(),
                            uniforms_sha256=hashlib.sha256(u.cpu().numpy().tobytes()).hexdigest()))
                    if a.diagnostics and requested=='low':
                        g2=torch.Generator(device=a.device).manual_seed(seed_for(common_seed,'latent-B'))
                        zb=torch.randn((a.count,latent_dim),device=a.device,generator=g2)
                        with torch.no_grad():xb=model.sample(c,n,z=zb,uniforms=u).cpu().numpy()
                        append(xb,entry,n,requested,'latent_B',common_seed,cksha,cfg,
                            extra=dict(latents_sha256=hashlib.sha256(zb.cpu().numpy().tobytes()).hexdigest(),
                                uniforms_sha256=hashlib.sha256(u.cpu().numpy().tobytes()).hexdigest()))
                else:
                    sampler_seed=seed_for(a.namespace,entry['method'],rep,n,requested)
                    rng=np.random.default_rng(sampler_seed)
                    if entry['sampler']=='uniform':
                        x=np.zeros((a.count,48),np.uint8)
                        for r in x:r[rng.choice(48,n,replace=False)]=1
                        append(x,entry,n,requested,'primary',sampler_seed,cksha,cfg)
                    else:
                        indices=np.flatnonzero(train&(data['n_Li']==n)&(labels==requested))
                        chosen=rng.choice(indices,a.count,replace=True)
                        seeds=data['occupancy'][chosen].copy();x=seeds.copy();extra=[]
                        for r,j in zip(x,chosen):
                            k=int(rng.integers(1,4));remove=rng.choice(np.flatnonzero(r),k,False)
                            add=rng.choice(np.flatnonzero(1-r),k,False);r[remove]=0;r[add]=1
                            extra.append(dict(seed_structure_id=str(data['structure_ids'][j]),
                                seed_source_row=int(j),seed_source_sha256=data_sha,
                                swap_count=k,removed_sites='|'.join(map(str,remove)),
                                added_sites='|'.join(map(str,add))))
                        append(x,entry,n,requested,'primary',sampler_seed,cksha,cfg,seeds,extra)
            if model is not None and a.diagnostics and not entry.get('descriptor_range'):
                rng=np.random.default_rng(seed_for(a.namespace,n,'posterior-VAL'))
                ii=rng.choice(np.flatnonzero(np.isin(data['split'],['val','validation'])&(data['n_Li']==n)),16,False)
                xs=torch.tensor(data['occupancy'][ii],device=a.device,dtype=torch.float32)
                xs,_=canonicalize(xs,inv)
                vectors=[resolve_request(cfg,n,str(labels[i]),None,schema)[0] for i in ii]
                c=torch.tensor(np.asarray(vectors),device=a.device)
                gg=torch.Generator(device=a.device).manual_seed(seed_for(common_seed,'posterior'))
                eps=torch.randn((16,latent_dim),generator=gg,device=a.device)
                uu=torch.rand((16,48),generator=gg,device=a.device)
                with torch.no_grad():
                    mu,lv=model.encode(xs,c);zp=mu+torch.exp(.5*lv)*eps
                    xp=model.sample(c,n,z=zp,uniforms=uu).cpu().numpy()
                    xprior=model.sample(c,n,z=eps,uniforms=uu).cpu().numpy()
                extra=[dict(validation_target_id=str(data['structure_ids'][i]),
                    validation_target_energy_eV_atom=float(data['energy_eV_atom'][i]),
                    posterior_target_hamming=int(np.abs(xp[j]-xs[j].cpu().numpy()).sum()),
                    prior_target_hamming=int(np.abs(xprior[j]-xs[j].cpu().numpy()).sum())) for j,i in enumerate(ii)]
                for j in range(16):
                    target=xs[j:j+1].cpu().numpy().astype(np.uint8)
                    append(xp[j:j+1],entry,n,str(labels[ii[j]]),f'posterior_{j}',common_seed,cksha,cfg,seeds=target,extra=extra[j])
                    append(xprior[j:j+1],entry,n,str(labels[ii[j]]),f'prior_matched_{j}',common_seed,cksha,cfg,seeds=target,extra=extra[j])
        if ck and sha256(ck)!=cksha:raise ValueError('Checkpoint mutated during generation')
        print('GENERATED',entry['method'],entry.get('fit_seed',entry.get('replicate_id')),flush=True)
    codes=np.unique([int(r['canonical_mask'],2) for r in rows]);masks=((codes[:,None]>>np.arange(47,-1,-1))&1).astype(np.uint8)
    np.savez_compressed(a.output_dir/'candidates.npz',masks=masks,occupancy=1-masks,
        structure_ids=np.array([f'mask48_{int(c):012x}' for c in codes]),n_Li=48-masks.sum(1),
        n_atoms=192-masks.sum(1),parent_sha256=np.asarray(parent_sha))
    np.savez_compressed(a.output_dir/'raw_attempts.npz',occupancy=np.asarray(raw),seed_occupancy=np.asarray(seed_raw))
    write_rows(a.output_dir/'attempts_all.csv',rows)
    primary=[r for r in rows if r['diagnostic']=='primary'];write_rows(a.output_dir/'attempts.csv',primary)
    receipt=dict(status='FROZEN_BEFORE_SCORING',namespace=a.namespace,primary_attempts=len(primary),
        total_attempts=len(rows),diagnostic_attempts=len(rows)-len(primary),unique_geometries=len(codes),
        no_energy_filtering=True,common_random_draws='same fullz32 and uniform(site) streams within count/replicate',
        posterior_scope='Protected validation for development diagnostic only; no test rows',
        training_data_sha256=data_sha,
        manifest_sha256=sha256(a.manifest),source_sha256=sha256(__file__),checkpoints=frozen,
        outputs={n:sha256(a.output_dir/n) for n in ['attempts.csv','attempts_all.csv','candidates.npz','raw_attempts.npz']},
        started_unix=started,finished_unix=time.time())
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
