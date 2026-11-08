"""Synthetic end-to-end paired proposal/accounting tests; no teacher calls."""
import csv,importlib.util,itertools,json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from cathode_cvae.conditions import fit_schema,save_schema
from cathode_cvae.data import sha256
from cathode_cvae.learning.models import make_model
from cathode_cvae.learning.generate_mask import canonical_masks

S=Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"
def module(name):
    spec=importlib.util.spec_from_file_location("test_"+name,S/(name+".py"));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
gen=module("generate_controls");evaluate=module("evaluate_controls");trained=module("audit_trained_controls")
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def fixture(tmp_path):
    parent=tmp_path/"parent.npz";np.savez(parent,numbers=np.array([11]*6+[8]),positions_A=np.zeros((7,3)),
        cell_A=np.eye(3)*5,li_indices=np.arange(6),permutations=np.arange(6)[None])
    x=np.zeros((20,6),np.uint8)
    for i,indices in enumerate(itertools.combinations(range(6),3)):x[i,list(indices)]=1
    split=np.array(["train"]*14+["val"]*4+["test"]*2);data=tmp_path/"data.npz"
    np.savez(data,occupancy=x,split=split,structure_ids=np.array([f"s{i}" for i in range(20)]),energy_eV_atom=np.arange(20.),n_target=np.full(20,3))
    schema=fit_schema(np.full(20,3),np.arange(20.),split,n_sites=6);schema_path=tmp_path/"schema.json";save_schema(schema,schema_path)
    keys,_=canonical_masks(x[:14],np.arange(6)[None]);history=tmp_path/"history.json";history.write_text(json.dumps(keys.tolist()))
    entries=[]
    for kind,name,latent in [("ar_cvae","cvae",4),("ar_baseline","ar",0)]:
        folder=tmp_path/name;folder.mkdir();checkpoint=folder/"best_validation_beta_0p1.pt"
        model=make_model(kind,n_sites=6,condition_dim=4,latent_dim=latent)
        cfg=dict(model_format_version=3,kind=kind,n_sites=6,condition_dim=4,latent_dim=latent,hidden_dim=64,
            encoder_kind="dense",graph_context=False,graph=None,decoder_context_spec=None,condition_mode="multibin",condition_schema=schema,
            parent_sha256=sha256(parent),data_sha256=sha256(data),seed=123,energy_target="deletion",label_energy_protocol="synthetic",proposal_geometry_protocol="synthetic",beta_schedule={})
        torch.save(dict(configuration=cfg,state_dict=model.state_dict(),optimizer_updates=4000,epoch=50,selection_beta=.1),checkpoint)
        (folder/"summary.json").write_text(json.dumps(dict(optimizer_updates=4000)))
        entries.append(dict(method=name,fit_seed=123,replicate_id=123,checkpoint=str(checkpoint),expected_checkpoint_sha256=sha256(checkpoint)))
    entries.extend([dict(method=name,replicate_id=123,sampler=name) for name in ("uniform","mutation")])
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps(dict(status="FROZEN_BEFORE_GENERATION",namespace="synthetic-paired-v1",models=entries,counts=[3],
        requested_bins=["low","middle","high"],diagnostics=True,count_per_group=4,expected_primary_attempts=48,
        inputs={k:dict(sha256=sha256(p)) for k,p in dict(parent=parent,data=data,schema=schema_path,history=history).items()})))
    return parent,data,schema_path,history,manifest


def invoke(tmp_path,monkeypatch=None):
    parent,data,schema,history,manifest=fixture(tmp_path);out=tmp_path/"generated"
    gen.main(["--manifest",str(manifest),"--parent",str(parent),"--data",str(data),"--condition-schema",str(schema),"--history",str(history),
        "--output-dir",str(out),"--count","4","--diagnostics","--device","cpu","--threads","1"])
    return parent,data,schema,history,manifest,out


def test_generic_generation_pairs_counts_union_and_actual_edit_arrays(tmp_path):
    _,_,_,_,_,out=invoke(tmp_path)
    attempts=rows(out/"attempts.csv");allrows=rows(out/"attempts_all.csv")
    assert len(attempts)==48 and len(allrows)==52
    with np.load(out/"raw_attempts.npz") as z:raw={k:z[k] for k in z.files}
    assert np.all(raw["occupancy"].sum(1)==3)
    for method in ("cvae","ar","uniform","mutation"):
        selected=[r for r in attempts if r["method"]==method]
        for i in range(4):
            rr=[r for r in selected if int(r["attempt"])==i]
            assert len(rr)==3
            indices=[int(r["raw_row"]) for r in rr]
            assert all(np.array_equal(raw["uniforms"][indices[0]],raw["uniforms"][j]) for j in indices)
            assert all(np.array_equal(raw["latents"][indices[0]],raw["latents"][j]) for j in indices)
            if method=="uniform":assert all(np.array_equal(raw["occupancy"][indices[0]],raw["occupancy"][j]) for j in indices)
            if method=="mutation":
                assert len({r["swap_count"] for r in rr})==1
                for r,j in zip(rr,indices):assert np.count_nonzero(raw["occupancy"][j]!=raw["seed_occupancy"][j])==2*int(r["swap_count"])
    with np.load(out/"candidates.npz") as z:
        assert len(set(z["structure_ids"].astype(str)))==len(z["structure_ids"])
        assert np.all(z["n_atoms"]==4)
    with np.load(out/"raw_primary.npz") as z:assert len(z["global_raw_row"])==48
    with np.load(out/"raw_diagnostics.npz") as z:assert len(z["global_raw_row"])==4


def test_evaluation_keeps_all_attempts_and_fit_replicate_denominators(tmp_path):
    parent,data,schema,history,_,out=invoke(tmp_path)
    with np.load(out/"candidates.npz") as z:ids=z["structure_ids"].copy();atoms=z["n_atoms"].copy()
    scores=tmp_path/"scores.npz";np.savez(scores,structure_ids=ids,n_atoms=atoms,energy_eV=np.arange(len(ids))*4.)
    thresholds=tmp_path/"q20.csv";thresholds.write_text("teacher,n_Li,source,q20_eV_atom\nsynthetic,3,main_train,2.0\n")
    result=tmp_path/"evaluation"
    receipt=evaluate.main(["--generated",str(out),"--parent",str(parent),"--history",str(history),"--training-data",str(data),
        "--score",f"synthetic={scores}","--schema",f"synthetic={schema}","--fixed-thresholds",str(thresholds),"--output-dir",str(result)])
    assert receipt["status"]=="PASS"
    metrics=rows(result/"metrics.csv");assert len(metrics)==12 and all(int(r["attempts"])==4 for r in metrics)
    assert len(rows(result/"seed_metrics.csv"))==4
    pairs=rows(result/"paired_attempts.csv");assert all(r["actual_z_equal"]=="True" for r in pairs if r["contrast"]!="alternate_latent_minus_low")
    assert all(r["AR_uniforms_equal"]=="True" for r in pairs)


def test_failed_model_requests_keep_primary_and_diagnostic_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr(gen,"model_draw",lambda *a,**k:(None,"Synthetic unsupported request"))
    _,_,_,_,_,out=invoke(tmp_path)
    primary=rows(out/"attempts.csv");allrows=rows(out/"attempts_all.csv")
    assert len(primary)==48 and len(allrows)==52
    assert sum(r["status"]=="failed_request" for r in primary)==24
    assert sum(r["status"]=="failed_request" for r in allrows)==28


def test_actual_checkpoint_audit_handles_no_encoder_control_without_energy_labels(tmp_path):
    parent,data,_,_,manifest=fixture(tmp_path);out=tmp_path/"audit.json"
    receipt=trained.main(["--manifest",str(manifest),"--data",str(data),"--parent",str(parent),"--output",str(out),"--threads","1"])
    assert receipt["status"]=="PASS" and not receipt["energy_labels_loaded"]
    assert len(receipt["models"])==2 and all(r["encoder_calls"]==0 for r in receipt["models"])


def test_missing_prespecified_score_is_unknown_and_never_a_complete_replay(tmp_path):
    parent,data,schema,history,_,out=invoke(tmp_path)
    with np.load(out/"candidates.npz") as z:ids=z["structure_ids"][:-1].copy();atoms=z["n_atoms"][:-1].copy()
    scores=tmp_path/"partial.npz";np.savez(scores,structure_ids=ids,n_atoms=atoms,energy_eV=np.arange(len(ids))*4.)
    result=tmp_path/"incomplete"
    with pytest.raises(SystemExit):
        evaluate.main(["--generated",str(out),"--parent",str(parent),"--history",str(history),"--training-data",str(data),
            "--score",f"synthetic={scores}","--schema",f"synthetic={schema}","--output-dir",str(result)])
    assert json.loads((result/"receipt.json").read_text())["status"]=="INCOMPLETE_COVERAGE"
    assert len(rows(result/"attempt_outcomes.csv"))==52


@pytest.mark.parametrize('provenance,complete',[('cached',True),('unverified',False)])
def test_collector_certified_cached_energies_are_measured_without_new_physics(tmp_path,provenance,complete):
    parent,data,schema,history,_,out=invoke(tmp_path)
    with np.load(out/'candidates.npz') as z:ids=z['structure_ids'].copy();atoms=z['n_atoms'].copy()
    scores=tmp_path/'cached.npz'
    np.savez(scores,structure_ids=ids,n_atoms=atoms,energy_eV=np.arange(len(ids))*4.,
        status=np.full(len(ids),'cached'),provenance=np.full(len(ids),provenance))
    result=tmp_path/'evaluated_cache'
    args=['--generated',str(out),'--parent',str(parent),'--history',str(history),'--training-data',str(data),
        '--score',f'synthetic={scores}','--schema',f'synthetic={schema}','--output-dir',str(result)]
    if complete:
        receipt=evaluate.main(args)
        assert receipt['status']=='PASS' and receipt['missing_or_nonterminal_candidate_labels']=={'synthetic':0}
        assert all(r['energy_known']=='True' and r['score_status']=='cached' for r in rows(result/'attempt_outcomes.csv'))
    else:
        with pytest.raises(SystemExit):evaluate.main(args)
        assert json.loads((result/'receipt.json').read_text())['status']=='INCOMPLETE_COVERAGE'
        assert all(r['energy_known']=='False' for r in rows(result/'attempt_outcomes.csv'))
    assert json.loads((result/'receipt.json').read_text())['new_MLIP_calls']==0


def test_sampler_metadata_honors_explicit_endpoint_label_target(tmp_path):
    parent,data,schema,history,manifest=fixture(tmp_path)
    value=json.loads(manifest.read_text())
    for entry in value['models']:
        if entry.get('sampler')=='mutation':entry['energy_target']='endpoint'
    manifest.write_text(json.dumps(value));out=tmp_path/'generated_endpoint_metadata'
    gen.main(['--manifest',str(manifest),'--parent',str(parent),'--data',str(data),
        '--condition-schema',str(schema),'--history',str(history),'--output-dir',str(out),
        '--count','4','--diagnostics','--device','cpu','--threads','1'])
    rr=rows(out/'attempts.csv')
    assert all(r['energy_target']=='endpoint' for r in rr if r['method']=='mutation')
    assert all(r['energy_target']=='deletion' for r in rr if r['method']!='mutation')
    assert all('unrelaxed' in r['geometry_protocol'] for r in rr)
