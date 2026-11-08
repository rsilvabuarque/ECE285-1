"""Synthetic engineering checks; no project training or physical labels."""
from datetime import datetime,timezone
import json

import numpy as np
import pytest

from cathode_cvae.conditions import fit_schema,classify
from cathode_cvae.learning.train_edit import main,validate_pairs,train_weights,digest
from cathode_cvae.learning.generate_edit import load_edit_model


def fixture_pairs():
    seeds=np.zeros((4,6),np.uint8);children=np.zeros_like(seeds)
    for i,(present,removed,added) in enumerate([([0,1,2],2,4),([0,1,3],3,5),([0,2,3],3,4),([1,2,3],3,5)]):
        seeds[i,present]=1;children[i]=seeds[i];children[i,removed]=0;children[i,added]=1
    key=lambda rows:np.asarray(["".join(map(str,1-r)) for r in rows])
    return dict(seeds=seeds,children=children,n_Li=np.full(4,3),swap_count=np.ones(4,int),
        split=np.asarray(["train","train","val","heldout"]),component_ids=np.asarray(["A","B","C","D"]),
        pair_ids=np.asarray(["a","b","c","d"]),canonical_seed_keys=key(seeds),canonical_child_keys=key(children))


def test_shared_identity_and_connected_components_cannot_bridge_roles():
    d=fixture_pairs();assert validate_pairs(d)["identity_role_bridges"]==0
    d["canonical_seed_keys"][2]=d["canonical_seed_keys"][0]
    with pytest.raises(ValueError,match="crosses editor roles"):validate_pairs(d)


def test_class_balancing_uses_train_children_and_preserves_val_weights():
    n=np.full(7,3);k=np.ones(7,int);b=np.array(["low","low","low","high","middle","high","low"])
    roles=np.array(["train"]*4+["val","heldout","heldout"])
    w,support=train_weights(n,k,b,roles)
    assert np.allclose(w[:3],2/3) and w[3]==2
    assert np.allclose(w[4:],1)
    assert {(r["child_bin"],r["train_rows"]) for r in support}=={("low",3),("high",1)}


def test_two_update_synthetic_pipeline_serializes_optional_phase(tmp_path):
    # This tiny engineering fit is deliberately outside the campaign run tree.
    # Its gate/report are synthetic fixtures, not an assertion about class work.
    parent=tmp_path/"synthetic_parent.npz";np.savez(parent,permutations=np.arange(6)[None])
    d=fixture_pairs();energy=np.array([-2.,-1.,-1.5,-1.7])
    schema=fit_schema(d["n_Li"],energy,d["split"],n_sites=6,
        provenance=dict(parent_sha256=digest(parent),data_sha256="synthetic-H0"))
    schema_path=tmp_path/"schema.json";schema_path.write_text(json.dumps(schema))
    d.update(child_energy_eV_atom=energy,child_bin=classify(schema,d["n_Li"],energy),
        parent_sha256=np.asarray(digest(parent)),H0_sha256=np.asarray("synthetic-H0"),condition_schema_id=np.asarray(schema["identity"]))
    data=tmp_path/"edit_pairs.npz";np.savez(data,**d)
    pdf=tmp_path/"synthetic_report.pdf";pdf.write_bytes(b"synthetic gate fixture; not a research report")
    gate=tmp_path/"gate.json";gate.write_text(json.dumps(dict(status="COMPLETE",completed_UTC="2020-01-01T00:00:00+00:00",
        report_pdf=dict(path=pdf.name,sha256=digest(pdf)))))
    prepared=tmp_path/"preparation_receipt.json";prepared.write_text(json.dumps(dict(status="READY_FOR_BOUNDED_EDITOR",
        identity_role_bridges=0,outputs={data.name:digest(data)},required_report_gate=dict(sha256=digest(gate)))))
    out=tmp_path/"engineering_fit"
    summary=main(["--root",str(tmp_path),"--data",str(data),"--preparation-receipt",str(prepared),"--parent",str(parent),
        "--condition-schema-in",str(schema_path),"--required-gate",str(gate),"--required-gate-sha256",digest(gate),
        "--seed","5","--counts","3","--latent-dim","2","--hidden-dim","8","--orientation","original",
        "--max-updates","2","--epochs","5","--batch-size","2","--beta-warmup","1","--validation-samples","1",
        "--class-balance","--device","cpu","--threads","1","--output-dir",str(out)])
    assert summary["status"]=="FIT_COMPLETE" and summary["optimizer_updates"]==2
    model,cfg=load_edit_model(out/"best_validation_beta_0p1.pt")
    assert cfg["kind"]=="seed_edit_cvae" and cfg["experiment_phase"]=="optional_editor"
    assert cfg["trainer_module"]=="cathode_cvae.learning.train_edit" and cfg["latent_dim"]==2
    assert cfg["heldout_rows"]==1 and summary["heldout_seed_families_evaluated"] is False
    assert summary["source_inputs_unchanged"] is True
