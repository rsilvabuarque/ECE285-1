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
