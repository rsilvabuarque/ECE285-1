"""Hash selection spans the whole TRAIN role, independent of endpoint outcomes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"))
import paired_common as common
import paired_prepare as prepare
import paired_mace_selection as selection
import paired_export_endpoints as export
import paired_worker as worker
from test_paired_relaxation import parent,Harmonic


def test_hash_selection_is_full_train_frozen_and_exportable_beyond_pilot(tmp_path):
    pp=tmp_path/"parent.npz";np.savez(pp,**parent())
    h=tmp_path/"history.json";h.write_text("[]")
    protected=tmp_path/"protected.npz";np.savez(protected,canonical_masks=np.array([],dtype="U4"),split=np.array([],dtype="U5"))
    protocol=tmp_path/"protocol.json";protocol.write_text('{"status":"synthetic"}')
    prepared=tmp_path/"prepared"
    prepare.prepare(SimpleNamespace(output_dir=prepared,parent=pp,history=h,protected_data=protected,
        goal_protocol=protocol,counts=[2],train_per_count=4,val_per_count=1,test_per_count=1,
        pilot_per_count=1,max_draws_per_count=1000,seed=314159))
    a=SimpleNamespace(prepared=prepared,output_dir=tmp_path/"selection",per_count=2,seed=12345)
    frozen=selection.freeze(a)
    a.output_dir=tmp_path/"repeat";again=selection.freeze(a)
    assert frozen["selected"]==again["selected"]
    assert frozen["eligible_train_per_count"]=={"2":4}
    assert not frozen["energies_read"] and not frozen["endpoint_files_read"]
    cohort=common.load_npz(prepared/"cohort.npz")
    selected=[r["cohort_row"] for r in frozen["selected"]]
    assert set(selected)-set(cohort["pilot_indices"])
    assert all(cohort["split"][row]=="train" for row in selected)
    tasks=json.loads((prepared/"tasks_all.json").read_text())["tasks"]
    context=json.loads((prepared/"context.json").read_text())
    queue=tmp_path/"queue";(queue/"results").mkdir(parents=True)
    from cathode_cvae.data import deletion_atoms
    for row in selected:
        task=tasks[row];atoms,_=deletion_atoms(parent(),cohort["occupancy"][row])
        target=atoms.positions.copy();target[-1,0]+=.1
        worker.run_attempt(Harmonic(target),parent(),task,context,queue/"results"/task["task_id"])
    result=export.export(SimpleNamespace(prepared=prepared,queue=queue,output_dir=tmp_path/"export",
        per_count=2,selection_manifest=tmp_path/"selection/selection.json"))
    assert result["exported"]==2 and result["selection"]=="Frozen full-TRAIN hash selection"
    assert {r["cohort_row"] for r in result["records"]}==set(selected)
    manifest=tmp_path/"selection/selection.json"
    bad=json.loads(manifest.read_text());bad["selection_seed"]+=1;manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="changed"):
        export.export(SimpleNamespace(prepared=prepared,queue=queue,output_dir=tmp_path/"tampered",
            per_count=2,selection_manifest=manifest))
