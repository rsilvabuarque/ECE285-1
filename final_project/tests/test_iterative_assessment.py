"""Synthetic assessments; all geometry and energies are artificial."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from cathode_cvae.data import sha256
from cathode_cvae.iterative_boundary import binding, freeze_manifest
from cathode_cvae.iterative_assessment import filter_assessment
from test_iterative_acquisition import fixture_data, write_json


def make_bundle(d):
    root=d['root'];boundary=root/'boundary';boundary.mkdir()
    write_json(boundary/'receipt.json',{'status':'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY',
      'parent':binding(d['parent']),'history':binding(d['known']),'protected':binding(d['protected'])})
    prior=root/'previous.json';write_json(prior,{'namespace':'before'})
    plan=root/'plan.json';write_json(plan,{'status':'DECLARED_ITERATIVE_DEVELOPMENT','purpose':'assessment',
      'namespace':'after','previous_manifests':[str(prior)],'counts':[24],'requested_bins':['low'],
      'count_per_group':4,'models':[{'method':'uniform','replicate_id':7,'sampler':'uniform'}]})
    manifest=root/'manifest.json';freeze_manifest(plan,boundary,d['parent'],d['base'],d['schema'],1,None,manifest)
    m=json.loads(manifest.read_text()); gen=root/'raw';gen.mkdir()
    inputs={n:binding(p) for n,p in [('parent',d['parent']),('data',d['base']),('schema',d['schema']),('history',d['known'])]}
    write_json(gen/'protocol.json',{'status':'FROZEN_BEFORE_GENERATION_AND_SCORING','no_energy_filtering':True,
      'namespace':'after','manifest':binding(manifest),'inputs':inputs,'condition_schema_id':m['expected_condition_schema_id'],
      'expected_primary_attempts':4,'count_per_group':4})
    occ=np.asarray([d['masks'][0],d['masks'][1],d['masks'][12],d['masks'][13]])
    rows=[]
    for i,x in enumerate(occ):
        key=''.join(map(str,1-x));sid=f'mask48_{int(key,2):012x}'
        rows.append(dict(attempt_id=f'id{i}',method='uniform',replicate_id='7',fit_seed='7',n_target='24',requested_bin='low',
                         canonical_mask=key,scoring_structure_id=sid,raw_row=str(i),status='generated',error='',
                         diagnostic='primary',paired_id=f'p{i}',attempt=str(i),latent_dim='0'))
    for name in ('attempts.csv','attempts_all.csv'):
        with (gen/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    raw={'occupancy':occ,'latents':np.zeros((4,8)), 'uniforms':np.arange(4*48).reshape(4,48)/200}
    np.savez_compressed(gen/'raw_attempts.npz',**raw)
    np.savez_compressed(gen/'raw_primary.npz',**raw)
    np.savez_compressed(gen/'raw_diagnostics.npz',**{k:v[:0] for k,v in raw.items()})
    for name in ('candidates.npz','primary_candidates.npz','diagnostic_candidates.npz'):
        np.savez_compressed(gen/name,placeholder=np.arange(2))
    generator=root/'generator.py';generator.write_text('# synthetic generator source\n')
    loader=root/'loader.py';loader.write_text('# synthetic loader source\n')
    outputs={p.name:sha256(p) for p in gen.iterdir() if p.name!='protocol.json'}
    write_json(gen/'receipt.json',{'status':'FROZEN_BEFORE_SCORING','no_energy_filtering':True,
      'protocol_sha256':sha256(gen/'protocol.json'),'source_sha256':sha256(generator),'runtime_loader_source':binding(loader),
      'outputs':outputs,'primary_attempts':4,'total_attempts':4})
    return boundary,manifest,gen,generator


def test_excludes_known_and_protected_preserves_arrays_and_evaluates(fixture_data,monkeypatch):
    d=fixture_data;b,m,g,s=make_bundle(d);out=d['root']/'filtered'
    result=filter_assessment(g,b,s,out)
    assert result['boundary_exclusions']=={'known':1,'protected':1}
    assert result['primary_attempts']==4
    assert sha256(out/'raw_attempts.npz')==sha256(g/'raw_attempts.npz')
    with np.load(out/'candidates.npz') as z:
        ids=z['structure_ids'];assert len(ids)==2
        n=z['n_target'];atoms=z['n_atoms']
    scores=d['root']/'scores.npz';np.savez_compressed(scores,structure_ids=ids,n_atoms=atoms,n_Li=n,
        energy_eV=np.full(2,-30.),status=np.asarray(['success']*2))
    script=Path(__file__).resolve().parents[1]/'scripts/overnight_20261108/evaluate_controls.py'
    spec=importlib.util.spec_from_file_location('assessment_evaluator',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'require_compute_allocation',lambda:None)
    evaluation=mod.main(['--generated',str(out),'--parent',str(d['parent']),'--history',str(d['known']),
      '--training-data',str(d['base']),'--output-dir',str(d['root']/'evaluation'),
      '--score',f'uma={scores}','--schema',f'uma={d["schema"]}'])
    assert evaluation['status']=='PASS' and evaluation['primary_attempts']==4


def test_manifest_tampering_and_missing_attempts_rejected(fixture_data):
    d=fixture_data;b,m,g,s=make_bundle(d)
    m.write_text(m.read_text()+' ')
    with pytest.raises(ValueError,match='hash mismatch'):filter_assessment(g,b,s,d['root']/'out1')
    b,m,g,s=make_bundle_new(d)
    rows=list(csv.DictReader((g/'attempts_all.csv').open()))[:-1]
    with (g/'attempts_all.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    r=json.loads((g/'receipt.json').read_text());r['outputs']['attempts_all.csv']=sha256(g/'attempts_all.csv')
    (g/'receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='denominator'):filter_assessment(g,b,s,d['root']/'out2')


def make_bundle_new(d):
    # Separate fresh paths; preserve the deliberately tampered first bundle.
    sub=d['root']/'second';sub.mkdir();d=dict(d,root=sub)
    return make_bundle(d)


def test_acquired_data_checkpoint_requires_matching_merge_and_generation_receipt(fixture_data):
    d=fixture_data;b,_,g,s=make_bundle(d);root=d['root']
    with np.load(d['base']) as z: arrays={k:z[k] for k in z.files}
    arrays['weight']=arrays['weight']*2
    merged=root/'training.npz';np.savez_compressed(merged,**arrays)
    merge=root/'merge.json';write_json(merge,{'status':'MERGED_ITERATIVE_ROUND','round':1,
       'outputs':{'training.npz':sha256(merged)},'origin_data_sha256':sha256(d['base']),
       'inputs':{'parent':binding(d['parent']),'schema':binding(d['schema'])}})
    checkpoint=root/'model.pt';checkpoint.write_bytes(b'synthetic checkpoint; never loaded')
    schema=json.loads(d['schema'].read_text())
    config=root/'configuration.json';write_json(config,{'seed':7,'parent_sha256':sha256(d['parent']),
       'data_sha256':sha256(merged),'condition_schema_id':schema['identity'],'condition_mode':'multibin',
       'energy_target':'deletion','latent_dim':8,'model_format_version':5})
    prior=root/'prior2.json';write_json(prior,{'namespace':'earlier'})
    plan=root/'plan2.json';write_json(plan,{'status':'DECLARED_ITERATIVE_DEVELOPMENT','purpose':'assessment',
       'namespace':'later','previous_manifests':[str(prior)],'counts':[24],'requested_bins':['low'],'count_per_group':4,
       'models':[{'method':'cvae','replicate_id':7,'fit_seed':7,'checkpoint':str(checkpoint)}]})
    manifest=root/'manifest2.json';freeze_manifest(plan,b,d['parent'],merged,d['schema'],2,merge,manifest)
    protocol=json.loads((g/'protocol.json').read_text());protocol['manifest']=binding(manifest)
    protocol['namespace']='later';protocol['inputs']['data']=binding(merged)
    (g/'protocol.json').write_text(json.dumps(protocol))
    for name in ('attempts.csv','attempts_all.csv'):
        with (g/name).open(newline='') as f: rows=list(csv.DictReader(f))
        for row in rows: row['method']='cvae';row['latent_dim']='8'
        with (g/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    receipt=json.loads((g/'receipt.json').read_text());receipt['protocol_sha256']=sha256(g/'protocol.json')
    receipt['checkpoints']=[{'method':'cvae','replicate_id':7,'actual_checkpoint_sha256':sha256(checkpoint)}]
    for name in ('attempts.csv','attempts_all.csv'): receipt['outputs'][name]=sha256(g/name)
    (g/'receipt.json').write_text(json.dumps(receipt))
    assert filter_assessment(g,b,s,root/'merged_filtered')['primary_attempts']==4
    config.write_text(config.read_text()+' ')
    with pytest.raises(ValueError,match='hash mismatch'):
        filter_assessment(g,b,s,root/'tampered_config')


def test_fixed_seed_mutation_bundle_passes_assessment_adapter(fixture_data):
    from cathode_cvae.fixed_seed_mutation import freeze_bank, generate
    import cathode_cvae.fixed_seed_mutation as source
    d=fixture_data;b,_,_,_=make_bundle(d);root=d['root']
    bank=root/'bank';freeze_bank(d['parent'],d['base'],d['schema'],d['protected'],[24],['low'],bank)
    old=root/'old_manifest.json';write_json(old,{'namespace':'earlier'})
    plan=root/'mutation_plan.json';write_json(plan,{'status':'DECLARED_ITERATIVE_DEVELOPMENT','purpose':'assessment',
      'namespace':'mutation-test','previous_manifests':[str(old)],'counts':[24],'requested_bins':['low'],
      'count_per_group':8,'models':[{'method':'mutation','replicate_id':7,'sampler':'mutation'}]})
    m=root/'mutation_manifest.json';freeze_manifest(plan,b,d['parent'],d['base'],d['schema'],1,None,m)
    generated=root/'mutation_raw';generate(m,d['parent'],d['base'],d['schema'],d['known'],d['protected'],bank,generated)
    result=filter_assessment(generated,b,source.__file__,root/'mutation_filtered')
    assert result['primary_attempts']==8
    assert sha256(generated/'raw_attempts.npz')==sha256(root/'mutation_filtered'/'raw_attempts.npz')
