"""Synthetic protocol and oracle checks; no model or scheduler jobs."""
import json
from pathlib import Path
import pytest
from cathode_cvae.iterative_boundary import binding
from cathode_cvae.iterative_comparison import freeze,_check_common,collect_oracle
from test_iterative_acquisition import fixture_data,generation,run_stage,oracle,write_json


def test_common_protocol_is_frozen_and_rejects_tampering(fixture_data):
    d=fixture_data;root=d['root'];b=root/'boundary';b.mkdir()
    write_json(b/'receipt.json',{'status':'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY','parent':binding(d['parent']),
       'history':binding(d['known']),'protected':binding(d['protected']),'request_schema':binding(d['schema'])})
    threshold=root/'threshold.csv';threshold.write_text('teacher,n_Li,q20_eV_atom\numa,24,-2\n')
    generator=root/'generator.py';generator.write_text('# generator\n')
    evaluator=root/'evaluator.py';evaluator.write_text('# evaluator\n')
    checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'synthetic')
    plan=root/'plan.json';write_json(plan,{'status':'DECLARED_MULTI_LINEAGE_DEVELOPMENT','teacher':'uma',
       'namespace':'new-shared','counts':[24],'requested_bins':['low'],'count_per_group':7,
       'condition_schema':str(d['schema']),'teacher_schema':str(d['schema']),'thresholds':str(threshold),
       'generator_source':str(generator),'evaluator_source':str(evaluator),
       'lineages':[{'policy':'A','replicate_id':7,'rng_kind':'ar','data':str(d['base']),'checkpoint':str(checkpoint)},
                   {'policy':'B','replicate_id':7,'rng_kind':'ar','data':str(d['base']),'checkpoint':str(checkpoint)}],
       'contrasts':[{'policy':'A','reference':'B'}]})
    out=root/'common.json';r=freeze(plan,b,out)
    assert r['status']=='FROZEN_MULTI_LINEAGE_BEFORE_GENERATION'
    assert len(_check_common(out,b)['lineages'])==2
    threshold.write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):_check_common(out,b)


def test_raw_fresh_oracle_union_and_cache_rejected(fixture_data):
    d=fixture_data;stage=d['root']/'stage';run_stage(d,generation(d,d['base']),stage)
    root,collection,_=oracle(d,stage)
    reg=json.loads((root/'registry.json').read_text())
    ids={v['structure_id'] for v in reg['physical_keys'].values()}
    created=json.loads((stage/'stage_receipt.json').read_text())['created_unix']
    result=collect_oracle(root,collection,ids,reg['parent_sha256'],'uma',created)
    assert result['unique_geometry_teacher_pairs']==2
    with pytest.raises(ValueError,match='eligible union'):
        collect_oracle(root,collection,ids|{'mask48_000000000000'},reg['parent_sha256'],'uma',created)
    for v in reg['physical_keys'].values():v['provenance']='exact_geometry_cache'
    (root/'registry.json').write_text(json.dumps(reg))
    cr=json.loads(collection.read_text())
    from cathode_cvae.data import sha256
    cr['registry_sha256']=sha256(root/'registry.json');collection.write_text(json.dumps(cr))
    with pytest.raises(ValueError,match='cached'):
        collect_oracle(root,collection,ids,reg['parent_sha256'],'uma',created)


def test_end_to_end_two_lineages_same_stream_and_fresh_oracle(fixture_data,monkeypatch):
    import csv,importlib.util,shutil
    import numpy as np
    from cathode_cvae.data import sha256
    from cathode_cvae.iterative_boundary import freeze_manifest
    from cathode_cvae.iterative_assessment import filter_assessment
    from cathode_cvae.iterative_comparison import aggregate
    from test_iterative_assessment import make_bundle
    d=fixture_data;root=d['root'];boundary,_,template,generator=make_bundle(d)
    threshold=root/'threshold.csv';threshold.write_text('teacher,n_Li,q20_eV_atom,source\numa,24,-2,main_train\n')
    evaluator=Path(__file__).resolve().parents[1]/'scripts/overnight_20261108/evaluate_controls.py'
    checkpoint=root/'model.pt';checkpoint.write_bytes(b'synthetic')
    schema=json.loads(d['schema'].read_text());config=root/'configuration.json'
    write_json(config,{'seed':7,'parent_sha256':sha256(d['parent']),'data_sha256':sha256(d['base']),
       'condition_schema_id':schema['identity'],'condition_mode':'multibin','energy_target':'deletion',
       'latent_dim':8,'model_format_version':5})
    plan=root/'common_plan.json';write_json(plan,{'status':'DECLARED_MULTI_LINEAGE_DEVELOPMENT','teacher':'uma',
       'namespace':'shared','counts':[24],'requested_bins':['low'],'count_per_group':4,
       'condition_schema':str(d['schema']),'teacher_schema':str(d['schema']),'thresholds':str(threshold),
       'generator_source':str(generator),'evaluator_source':str(evaluator),
       'lineages':[{'policy':q,'replicate_id':7,'rng_kind':'ar','data':str(d['base']),'checkpoint':str(checkpoint)} for q in ('A','B')],
       'contrasts':[{'policy':'A','reference':'B'}]})
    common=root/'common.json';freeze(plan,boundary,common)
    earlier=root/'prior_x.json';write_json(earlier,{'namespace':'earlier'})
    safe_dirs=[]
    for policy in ('A','B'):
        lp=root/f'plan_{policy}.json';write_json(lp,{'status':'DECLARED_ITERATIVE_DEVELOPMENT','purpose':'assessment',
          'common_protocol_sha256':sha256(common),'namespace':'shared','previous_manifests':[str(earlier)],
          'counts':[24],'requested_bins':['low'],'count_per_group':4,
          'models':[{'method':policy,'replicate_id':7,'fit_seed':7,'checkpoint':str(checkpoint)}]})
        mf=root/f'manifest_{policy}.json';freeze_manifest(lp,boundary,d['parent'],d['base'],d['schema'],1,None,mf)
        raw=root/f'raw_{policy}';shutil.copytree(template,raw)
        proto=json.loads((raw/'protocol.json').read_text());proto['manifest']=binding(mf);proto['namespace']='shared'
        (raw/'protocol.json').write_text(json.dumps(proto))
        for name in ('attempts.csv','attempts_all.csv'):
            with (raw/name).open(newline='') as f: rows=list(csv.DictReader(f))
            for row in rows:row['method']=policy;row['latent_dim']='8'
            with (raw/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        receipt=json.loads((raw/'receipt.json').read_text());receipt['protocol_sha256']=sha256(raw/'protocol.json')
        receipt['checkpoints']=[{'method':policy,'replicate_id':7,'actual_checkpoint_sha256':sha256(checkpoint)}]
        for name in ('attempts.csv','attempts_all.csv'):receipt['outputs'][name]=sha256(raw/name)
        (raw/'receipt.json').write_text(json.dumps(receipt))
        safe=root/f'safe_{policy}';filter_assessment(raw,boundary,generator,safe);safe_dirs.append(safe)
    after=max(json.loads((p/'receipt.json').read_text())['created_unix'] for p in safe_dirs)
    write_json(safe_dirs[0]/'stage_receipt.json',{'created_unix':after})
    queue,collection,_=oracle(d,safe_dirs[0])
    spec=importlib.util.spec_from_file_location('comparison_evaluator',evaluator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'require_compute_allocation',lambda:None)
    bundle_args=[]
    for policy,safe in zip(('A','B'),safe_dirs):
        output=root/f'eval_{policy}'
        module.main(['--generated',str(safe),'--parent',str(d['parent']),'--history',str(d['known']),
          '--training-data',str(d['base']),'--score',f'uma={queue}/collected/uma.npz','--schema',f'uma={d["schema"]}',
          '--fixed-thresholds',str(threshold),'--output-dir',str(output)])
        bundle_args.append(f'{policy}:7={safe}={output}')
    result=aggregate(common,boundary,bundle_args,queue,collection,root/'summary')
    assert result['global_oracle_calls']==2 and result['standalone_eligible_queries']==4
    with (root/'summary'/'paired_differences.csv').open(newline='') as f:row=next(csv.DictReader(f))
    assert float(row['balanced_difference'])==0 and float(row['low_difference'])==0
    raw_path=safe_dirs[1]/'raw_attempts.npz'
    with np.load(raw_path) as z: arr={k:z[k] for k in z.files}
    arr['uniforms'][0,0]+=0.125
    np.savez_compressed(raw_path,**arr)
    receipt=safe_dirs[1]/'receipt.json';changed=json.loads(receipt.read_text())
    changed['outputs']['raw_attempts.npz']=sha256(raw_path);receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='AR streams differ'):
        aggregate(common,boundary,bundle_args,queue,collection,root/'bad_summary')
