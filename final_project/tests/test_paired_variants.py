"""Frozen numerical controls and full-precision endpoint export; no MLIP calls."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"))
import paired_common as common
import paired_prepare as prepare
import paired_variants as variants
import paired_variant_worker as variant_worker
import paired_export_endpoints as export
import paired_worker as worker
import paired_compare_controls as compare
from test_paired_relaxation import parent,Harmonic


def prepared_fixture(tmp_path):
    pp=tmp_path/"parent.npz";np.savez(pp,**parent())
    h=tmp_path/"history.json";h.write_text('["0011"]')
    protected=tmp_path/"protected.npz";np.savez(protected,canonical_masks=["0011"],split=["test"])
    protocol=tmp_path/"protocol.json";protocol.write_text('{"status":"synthetic"}')
    output=tmp_path/"prepared"
    prepare.prepare(SimpleNamespace(output_dir=output,parent=pp,history=h,protected_data=protected,
        goal_protocol=protocol,counts=[2],train_per_count=2,val_per_count=1,test_per_count=1,
        pilot_per_count=2,max_draws_per_count=1000,seed=314159))
    return output


def test_tighter_and_secondary_teacher_contexts_preserve_roles_parent_and_original_bytes(tmp_path):
    prepared=prepared_fixture(tmp_path)
    before={p.name:common.sha256(p) for p in prepared.iterdir() if p.is_file()}
    out=tmp_path/"variants"
    variants.prepare_variants(SimpleNamespace(prepared=prepared,output_dir=out,tight_per_count=2,mace_per_count=1))
    _,tight,tasks=variant_worker.verify_variant(prepared,out/"tight_0p01")
    _,mace,mtasks=variant_worker.verify_variant(prepared,out/"mace_independent")
    assert tight["optimizer"]["fmax_eV_A"]==.01 and len(tasks)==2
    assert mace["teacher"]["model"]=="mace" and mace["optimizer"]["fmax_eV_A"]==.05 and len(mtasks)==1
    assert tight["geometry"]==mace["geometry"] and tight["parent_sha256"]==mace["parent_sha256"]
    assert all(t["split"]=="train" for t in tasks+mtasks)
    assert tasks[0]["physical_key"]!=mtasks[0]["physical_key"]
    assert before=={p.name:common.sha256(p) for p in prepared.iterdir() if p.is_file()}
    (out/"tight_0p01/context.json").write_text("{}")
    with pytest.raises(ValueError,match="changed"):
        variant_worker.verify_variant(prepared,out/"tight_0p01")


def test_endpoint_npz_export_retains_every_coordinate_bit_for_portable_mlip_input(tmp_path):
    from cathode_cvae.data import deletion_atoms
    from cathode_cvae.mlip.cli import read_inputs
    prepared=prepared_fixture(tmp_path)
    context=json.loads((prepared/"context.json").read_text())
    cohort=common.load_npz(prepared/"cohort.npz")
    tasks=json.loads((prepared/"tasks_all.json").read_text())["tasks"]
    queue=tmp_path/"queue";(queue/"results").mkdir(parents=True)
    expected={}
    for row in cohort["pilot_indices"]:
        task=tasks[int(row)];atoms,_=deletion_atoms(parent(),cohort["occupancy"][row])
        target=atoms.positions.copy();target[-1,0]+=.123456789123
        directory=queue/"results"/task["task_id"]
        record,_=worker.run_attempt(Harmonic(target),parent(),task,context,directory)
        arrays=common.load_npz(directory/"arrays.npz")
        expected[str(cohort["structure_ids"][row])]=arrays["endpoint_positions_A"]
        # Energy poison is retained in source bytes but never read by export.
        arrays["endpoint_energy_eV"]=np.array(np.nan)
        common.atomic_npz(directory/"arrays.npz",**arrays)
        record["arrays_sha256"]=common.sha256(directory/"arrays.npz")
        common.publish_result(directory,record)
    out=tmp_path/"export"
    receipt=export.export(SimpleNamespace(prepared=prepared,queue=queue,output_dir=out,per_count=2))
    assert receipt["exported"]==2 and not receipt["energy_labels_read"]
    loaded=list(read_inputs([out/"endpoints.npz"],prepared/"parent.npz"))
    assert len(loaded)==2
    for identity,atoms,source in loaded:
        assert np.array_equal(atoms.positions,expected[identity])


def test_control_comparison_keeps_tightening_and_mace_sp_distinct(tmp_path):
    from cathode_cvae.data import deletion_atoms
    from cathode_cvae.conditions import fit_schema,save_schema
    prepared=prepared_fixture(tmp_path)
    context=json.loads((prepared/"context.json").read_text())
    cohort=common.load_npz(prepared/"cohort.npz")
    tasks=json.loads((prepared/"tasks_all.json").read_text())["tasks"]
    variant_dir=tmp_path/"variants"
    variants.prepare_variants(SimpleNamespace(prepared=prepared,output_dir=variant_dir,tight_per_count=2,mace_per_count=1))
    primary=tmp_path/"primary";controls=tmp_path/"controls"
    for queue in [primary,controls]:(queue/"results").mkdir(parents=True)
    for row in cohort["pilot_indices"]:
        task=tasks[int(row)];atoms,_=deletion_atoms(parent(),cohort["occupancy"][row])
        target=atoms.positions.copy();target[-1,0]+=.1
        worker.run_attempt(Harmonic(target),parent(),task,context,primary/"results"/task["task_id"])
    for name in ["tight_0p01","mace_independent"]:
        _,vcontext,vtasks=variant_worker.verify_variant(prepared,variant_dir/name)
        for task in vtasks:
            atoms,_=deletion_atoms(parent(),cohort["occupancy"][task["input_row"]])
            target=atoms.positions.copy();target[-1,0]+=.15 if name=="mace_independent" else .1
            worker.run_attempt(Harmonic(target),parent(),task,vcontext,controls/"results"/task["task_id"])
    exported=tmp_path/"export"
    export.export(SimpleNamespace(prepared=prepared,queue=primary,output_dir=exported,per_count=2))
    manifest=json.loads((exported/"manifest.json").read_text())
    data=common.load_npz(exported/"endpoints.npz")
    mace=tmp_path/"mace_sp";mace.mkdir()
    model=json.loads((variant_dir/"mace_independent/context.json").read_text())["teacher"]
    common.atomic_json(mace/"model_receipt.json",dict(model,ase="3.26.0"))
    energies=[];forces=[]
    for i,record in enumerate(manifest["records"]):
        slots=common.load_npz(primary/"results"/record["primary_task_id"]/"arrays.npz")["parent_indices"]
        target=parent()["positions_A"][slots].copy();target[-1,0]+=.15
        lo,hi=data["offsets"][i:i+2];delta=data["positions"][lo:hi]-target
        energies.append(.5*np.sum(delta**2));forces.append(-delta)
    common.atomic_npz(mace/"predictions.npz",**{k:data[k] for k in ["structure_ids","offsets","numbers","positions","cell"]},
        energy_eV=np.array(energies),n_atoms=np.diff(data["offsets"]),forces_eV_A=np.concatenate(forces))
    (mace/"results.jsonl").write_text("".join(json.dumps(dict(structure_id=s,status="success"))+"\n" for s in data["structure_ids"]))
    schema=fit_schema([2]*5,[0.,.1,.2,.3,.4],["train"]*5,n_sites=4,
        provenance=dict(endpoint_context_hash=context["context_hash"]))
    schema_path=tmp_path/"schema.json";save_schema(schema,schema_path)
    result=compare.main(["--prepared",str(prepared),"--primary-queue",str(primary),"--variants",str(variant_dir),
        "--variant-queue",str(controls),"--endpoint-schema",str(schema_path),"--mace-sp-directory",str(mace),
        "--mace-sp-manifest",str(exported/"manifest.json"),"--output-dir",str(tmp_path/"comparison")])
    assert result["status"]=="PASS_COMPLETE" and result["controls"]==3
    import csv
    with (tmp_path/"comparison/mace_endpoint_checks.csv").open() as stream:rows=list(csv.DictReader(stream))
    independent=[r for r in rows if r["mace_independent_energy_eV_atom"]]
    assert len(independent)==1 and float(independent[0]["mace_relax_gain_from_uma_meV_atom"])<0
