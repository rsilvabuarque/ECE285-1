"""Final planning requires explicit independent fits and preserves TEST sealing."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"))
import paired_common as common
import paired_assessment_plan as plan
import paired_prepare as prepare
from cathode_cvae.conditions import fit_schema,save_schema
from test_paired_relaxation import parent


def test_plan_freezes_four_scopes_without_selecting_or_running_models(tmp_path):
    pp=tmp_path/"parent.npz";np.savez(pp,**parent())
    history0=tmp_path/"history.json";history0.write_text("[]")
    protected=tmp_path/"protected.npz";np.savez(protected,canonical_masks=np.array([],dtype="U4"),split=np.array([],dtype="U5"))
    goal=tmp_path/"protocol.json";goal.write_text('{"status":"synthetic"}')
    prepared=tmp_path/"prepared"
    prepare.prepare(SimpleNamespace(output_dir=prepared,parent=pp,history=history0,protected_data=protected,
        goal_protocol=goal,counts=[1,2,3],train_per_count=2,val_per_count=1,test_per_count=1,
        pilot_per_count=1,max_draws_per_count=1000,seed=314159))
    context=json.loads((prepared/"context.json").read_text())
    data=common.load_npz(prepared/"cohort.npz")
    # A three-composition fixture for planning arithmetic only, with explicit
    # occupied counts. No checkpoint in this software test is a neural model.
    rows=np.array([i for i,s in enumerate(data["split"]) if s!="test"])
    inputs={"parent":str(prepared/"parent.npz"),"paired_prepared":str(prepared)}
    history=tmp_path/"final_history.json";history.write_text(json.dumps(dict(keys=sorted(set(data["canonical_masks"])|{"0011"}))))
    inputs.update(history=str(history),intake_history=str(tmp_path/"history.json"),original_thresholds=str(tmp_path/"protocol.json"))
    history_receipt=tmp_path/"history_receipt.json"
    history_receipt.write_text(json.dumps(dict(status="FROZEN_IDENTITIES_ONLY",history_sha256=common.sha256(history),
        intake_history_sha256=common.sha256(tmp_path/"history.json"),cohort_sha256=common.sha256(prepared/"cohort.npz"),
        declared_exposure_manifest_verified=True)))
    inputs["history_receipt"]=str(history_receipt)
    for target in ["deletion","endpoint"]:
        path=tmp_path/(target+".npz")
        np.savez(path,**{k:data[k][rows] for k in ["occupancy","split","structure_ids","n_target","canonical_masks"]},
            weight=np.ones(len(rows)),energy_eV_atom=np.array([np.nan]*len(rows)))
        # NaN labels prove planning never loads/requires the energy column. The
        # real generator/trainer separately require their finite actual labels.
        inputs[f"paired_{target}_data"]=str(path)
        schema=fit_schema([1,1,1,2,2,2,3,3,3],[0.,1.,2.]*3,["train"]*9,n_sites=4,
            provenance=dict(energy_target=target,endpoint_context_hash=context["context_hash"]))
        sp=tmp_path/(target+"_schema.json");save_schema(schema,sp);inputs[f"paired_{target}_schema"]=str(sp)
    inputs["current_data"]=inputs["paired_deletion_data"];inputs["current_schema"]=inputs["paired_deletion_schema"]
    arms={}
    for name in plan.ARMS:
        arms[name]=[]
        for rep in range(3):
            target="endpoint" if name=="paired_endpoint_cvae" else "deletion"
            dname="paired_endpoint_data" if target=="endpoint" else "paired_deletion_data" if name=="paired_deletion_cvae" else "current_data"
            sid=json.loads(Path(inputs[dname.replace("_data","_schema")]).read_text())["identity"]
            cfg=dict(kind="ar_baseline" if name=="current_ar" else "ar_cvae",latent_dim=0 if name=="current_ar" else 4,
                graph_context=True,encoder_kind="dense" if name=="current_ar" else "graph",energy_target=target,seed=rep,
                data_sha256=common.sha256(inputs[dname]),parent_sha256=common.sha256(inputs["parent"]),
                condition_schema_id=sid,condition_mode="multibin")
            path=tmp_path/(name+str(rep)+".pt");torch.save(dict(configuration=cfg,state_dict={},synthetic_arm=name),path)
            arms[name].append(dict(checkpoint=str(path),fit_seed=rep,expected_checkpoint_sha256=common.sha256(path)))
    scripts=tmp_path/"scripts/overnight_20261108";scripts.mkdir(parents=True)
    for name in ["generate_controls.py","paired_register.py"]:(scripts/name).write_text("# synthetic non-executed source\n")
    for name in ["paired_common.py","paired_assessment_plan.py"]:
        (scripts/name).write_bytes((Path(plan.__file__).parent/name).read_bytes())
    selection=tmp_path/"selection.json"
    chosen=dict(status="FROZEN_DEVELOPMENT_SELECTION",namespace="synthetic-only",counts=[1,2,3],inputs=inputs,arms=arms)
    selection.write_text(json.dumps(chosen))
    args=SimpleNamespace(root=tmp_path,selection=selection,output_dir=tmp_path/"plan")
    result=plan.freeze(args)
    assert result["conservative_endpoint_budget"]["proposal_attempts"]==10944
    assert result["new_MLIP_calls"]==0 and result["no_model_selection_or_training_performed"]
    assert result["stages"]["current_low"]["attempts"]==2304
    endpoint=json.loads(Path(result["stages"]["paired_endpoint"]["manifest"]["path"]).read_text())
    assert all(r["energy_target"]=="endpoint" for r in endpoint["models"])
    uniform=json.loads(Path(result["stages"]["shared_uniform"]["manifest"]["path"]).read_text())
    current=json.loads(Path(result["stages"]["current_low"]["manifest"]["path"]).read_text())
    assert uniform["namespace"]==current["namespace"] and uniform["inputs"]["history"]==current["inputs"]["history"]
    plan.verify_plan(args.output_dir/"plan.json")
    Path(arms["current_cvae"][0]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        plan.verify_plan(args.output_dir/"plan.json")
    chosen["status"]="PENDING_SELECTION";selection.write_text(json.dumps(chosen));args.output_dir=tmp_path/"pending"
    with pytest.raises(ValueError,match="DEVELOPMENT selection"):
        plan.freeze(args)
