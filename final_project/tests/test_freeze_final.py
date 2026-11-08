"""Final-boundary invariants without touching actual assessment files."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/freeze_final.py"
spec = importlib.util.spec_from_file_location("final_freezer",SOURCE)
freezer = importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)


def test_generator_enforces_actual_frozen_manifest_keys(tmp_path):
    source = SOURCE.with_name("generate_study.py")
    module_spec = importlib.util.spec_from_file_location("frozen_generator", source)
    generator = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(generator)
    history = tmp_path/"history.json";history.write_text('["0001"]\n')
    manifest = dict(status="FROZEN_BEFORE_FINAL_GENERATION",expected_parent_sha256="parent",
        expected_training_data_sha256="data",expected_condition_schema_id="schema",namespace="final",
        count_per_composition_bin_replicate=128,diagnostics=False,models=[{},{}],
        counts=[12,24,36],requested_bins=["low","middle","high"],expected_primary_attempts=2304,
        history_path=str(history),history_sha256=freezer.digest(history))
    kwargs = dict(parent_sha="parent",data_sha="data",schema_id="schema",namespace="final",
                  count=128,diagnostics=False)
    generator.validate_frozen_manifest(manifest,**kwargs)
    for key, value in [("parent_sha","other"),("data_sha","other"),("schema_id","other"),
                       ("namespace","dev"),("count",64),("diagnostics",True)]:
        with pytest.raises(ValueError,match="Frozen final manifest"):
            generator.validate_frozen_manifest(manifest,**dict(kwargs,**{key:value}))
    history.write_text('["0010"]\n')
    with pytest.raises(ValueError,match="history changed"):
        generator.validate_frozen_manifest(manifest,**kwargs)


def test_primary_and_cross_teacher_budgets_match_replicated_families():
    rows = freezer.family_specs()
    assert len(rows) == 30
    assert len({(r["method"],r["replicate_id"]) for r in rows}) == 30
    assert len(rows)*3*3*128 == 34560
    matched = [r for r in rows if r["method"] in freezer.MACE_FAMILIES]
    assert len(matched) == 18 and len(matched)*3*3*128 == 20736
    context = [r for r in rows if r["method"] == "graph_context"]
    assert len(context) == 3 and all(r["encoder_kind"] == "graph" and r["graph_context"] for r in context)
    baseline = [r for r in rows if r["method"] == "ar_baseline"]
    assert len(baseline) == 3 and all(r["latent_dim"] == 0 for r in baseline)


def test_geometry_byte_encoding_preserves_sites_and_never_pads_another_cell():
    assert freezer.decode_audit_orbit(bytes([0,0,1,1]).hex(),4) == "0011"
    with pytest.raises(ValueError,match="exactly the parent site count"):
        freezer.decode_audit_orbit(bytes([0]*12).hex(),48)
    with pytest.raises(ValueError,match="exactly the parent site count"):
        freezer.decode_audit_orbit(bytes([0,0,2,1]).hex(),4)


def test_candidate_parent_count_and_canonicality_are_checked(tmp_path):
    permutations = np.asarray([[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]])
    p = tmp_path/"candidates.npz"
    masks = np.asarray([[0,0,1,1],[0,1,0,1]],np.uint8)
    def write(**extra):
        values=dict(masks=masks,occupancy=1-masks,structure_ids=["A","B"],n_Li=[2,2],
                    n_atoms=[4,4],parent_sha256=np.asarray("same-parent"),energy_eV=[-1e9,float("nan")])
        values.update(extra);np.savez(p,**values)
    write()
    keys, record = freezer.load_pool(p,"same-parent",4,permutations,2)
    assert keys == {"0011","0101"} and "energy_eV" not in record
    with pytest.raises(ValueError,match="another or unidentified parent"):
        freezer.load_pool(p,"other-parent",4,permutations,2)
    write(masks=np.asarray([[0,1,1,0],[0,1,0,1]]),occupancy=np.asarray([[1,0,0,1],[1,0,1,0]]))
    with pytest.raises(ValueError,match="not canonical"):
        freezer.load_pool(p,"same-parent",4,permutations,2)
    write()
    with pytest.raises(ValueError,match="without padding"):
        freezer.load_pool(p,"same-parent",48,np.arange(48)[None],144)


def test_geometry_history_uses_exact_deletion_masks_not_endpoint_labels(tmp_path):
    data=tmp_path/"geometry.npz"
    np.savez(data,occupancy=np.asarray([[1,1,0,0],[0,0,1,1]],np.uint8),
        structure_ids=np.asarray(["synthetic","synthetic"]),cohort=np.asarray(["deletion","fixed_endpoint"]),
        n_Li=np.asarray([2,2]),energy_eV=np.asarray([-1.,-99.]))
    csv_path=tmp_path/"lineages.csv"
    with csv_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["structure_id","n_Li","deletion_frame","initial_occupation_orbit"])
        w.writeheader();w.writerow(dict(structure_id="synthetic",n_Li=2,deletion_frame=0,
            initial_occupation_orbit=bytes([0,0,1,1]).hex()))
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps(dict(status="PASS",lineages=1,input_hashes={
        "core/data/parents/O3_N48_pilot.npz":"same-parent","core/data/geometry_uma_fixed.npz":freezer.digest(data)})))
    keys, record = freezer.geometry_keys(csv_path,receipt,data,"same-parent",4,np.arange(4)[None])
    assert keys == {"0011"} and "energies" in record["role"]
    assert "energy_eV" not in record


def test_publish_is_write_once_and_protocol_is_last(tmp_path,monkeypatch):
    run=tmp_path/"run";run.mkdir()
    history,manifest,protocol=run/"data/history.json",run/"manifest.json",run/"protocol.json"
    output={history:b"[]\n",manifest:b'{"models":[]}\n',protocol:b'{"frozen":true}\n'}
    order=[];real_link=freezer.os.link
    def recorded_link(source,target):
        order.append(target);return real_link(source,target)
    monkeypatch.setattr(freezer.os,"link",recorded_link)
    freezer.publish_once(run,output,protocol)
    assert order[-1] == protocol
    assert all(path.read_bytes() == value for path,value in output.items())
    with pytest.raises(FileExistsError):freezer.publish_once(run,{history:b"CHANGED",run/"new":b"new",protocol:b"new"},protocol)
    assert history.read_bytes() == b"[]\n" and not (run/"new").exists()


def test_incompatible_checkpoint_dimensions_fail_before_publication(tmp_path):
    run=tmp_path/"run";folder=run/"fits/latent_d4_20261106";folder.mkdir(parents=True)
    cfg=dict(model_format_version=2,kind="ar_cvae",n_sites=48,condition_mode="multibin",condition_dim=4,latent_dim=8)
    (folder/"configuration.json").write_text(json.dumps(cfg));(folder/"summary.json").write_text("{}")
    spec=next(r for r in freezer.family_specs() if r["method"] == "cvae_d4")
    with pytest.raises(ValueError,match="incompatible latent_dim"):
        freezer.checkpoint_entry(spec,run,"parent","data","split","schema",[])


def test_checkpoint_embedded_identity_and_selected_update_are_preserved(tmp_path):
    import torch
    from cathode_cvae.learning.models import make_model
    run=tmp_path/"run";folder=run/"fits/latent_d4_20261106";folder.mkdir(parents=True)
    s=next(r for r in freezer.family_specs() if r["method"] == "cvae_d4")
    cfg=dict(model_format_version=2,kind="ar_cvae",n_sites=48,condition_mode="multibin",condition_dim=4,
        latent_dim=4,hidden_dim=64,encoder_kind="dense",graph_context=False,orientation="canonical",seed=20261106,
        parent_sha256="parent",data_sha256="data",split_sha256="split",condition_schema_id="schema",
        train_rows=40157,val_rows=4967,test_rows=5123,counts=[12,24,36],teacher="uma",teacher_head="omat",
        dtype="float32",tf32=False,geometry_protocol=freezer.PROTOCOL,class_balance=True,beta_final=.1,
        maximum_updates=4000,count_likelihood=True,trainer_sha256="a"*64,model_source_sha256="b"*64)
    (folder/"configuration.json").write_text(json.dumps(cfg))
    checkpoint=folder/"best_validation_beta_0p1.pt"
    model=make_model("ar_cvae",n_sites=48,condition_dim=4,latent_dim=4)
    stored=dict(state_dict=model.state_dict(),configuration=cfg,selection_beta=.1,epoch=3,optimizer_updates=200)
    torch.save(stored,checkpoint)
    def summary():
        value=dict(status="FIT_COMPLETE",optimizer_updates=4000,heldout_test_evaluated=False,
            selection_checkpoints={"0.1":dict(filename=checkpoint.name,epoch=3,sha256=freezer.digest(checkpoint))})
        (folder/"summary.json").write_text(json.dumps(value))
    summary()
    tasklist=run/"tasks.json";tasklist.write_text("{}")
    claim=run/"tasks.claims"/folder.name;claim.mkdir(parents=True)
    (claim/"receipt.json").write_text(json.dumps(dict(status="COMPLETE",tasks_sha256=freezer.digest(tasklist))))
    snapshots=[dict(sha256="source-manifest",files={"cathode_cvae/learning/train_mask.py":"a"*64,
                                               "cathode_cvae/learning/models.py":"b"*64})]
    result=freezer.checkpoint_entry(s,run,"parent","data","split","schema",snapshots)
    assert result["selected_checkpoint_optimizer_updates"] == 200
    assert result["expected_checkpoint_sha256"] == freezer.digest(checkpoint)
    stored["configuration"]={**cfg,"parent_sha256":"wrong-parent"}
    torch.save(stored,checkpoint);summary()
    with pytest.raises(ValueError,match="embedded configuration disagrees.*parent_sha256"):
        freezer.checkpoint_entry(s,run,"parent","data","split","schema",snapshots)
