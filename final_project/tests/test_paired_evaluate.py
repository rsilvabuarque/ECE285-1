"""All-attempt endpoint metrics on a synthetic frozen proposal/label bundle."""
import csv
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/overnight_20261108"))
import paired_common as common
import paired_prepare as prepare
import paired_register as register
import paired_evaluate as evaluate
from cathode_cvae.conditions import fit_schema, save_schema


def test_duplicates_known_masks_and_failed_attempts_stay_in_endpoint_denominator(tmp_path):
    parent = tmp_path / "parent.npz"
    np.savez(parent, numbers=[8,8,3,3,3,3], positions_A=np.array([[0,0,0],[5,5,5],[1,2,2],[3,2,2],[5,2,2],[7,2,2]],float),
        cell_A=np.eye(3)*10, li_indices=[2,3,4,5], permutations=[[0,1,2,3]])
    history = tmp_path / "intake.json"; history.write_text('["0011"]')
    protected = tmp_path / "protected.npz"; np.savez(protected, canonical_masks=["0011"], split=["test"])
    protocol = tmp_path / "protocol.json"; protocol.write_text('{"status":"synthetic"}')
    prepared = tmp_path / "prepared"
    prepare.prepare(SimpleNamespace(output_dir=prepared, parent=parent, history=history,
        protected_data=protected, goal_protocol=protocol, counts=[2], train_per_count=2,
        val_per_count=1, test_per_count=1, pilot_per_count=2, max_draws_per_count=1000, seed=314159))
    context = json.loads((prepared / "context.json").read_text())
    cohort = common.load_npz(prepared / "cohort.npz")
    known = set(cohort["canonical_masks"]) | {"0011"}
    all_keys = {"".join(bits) for bits in itertools.product("01",repeat=4) if bits.count("0")==2}
    novel = (all_keys - known).pop()
    final_history = tmp_path / "final_history.json"; final_history.write_text(json.dumps(sorted(known)))
    pool = tmp_path / "pool.npz"
    np.savez(pool, occupancy=[common.occupancy_from_key("0011"),common.occupancy_from_key(novel)], structure_ids=["known","novel"])
    attempts = tmp_path / "attempts.csv"
    rows = [dict(method="synthetic",fit_seed="1",replicate_id="0",n_Li="2",requested_bin="low",status="success",
        scoring_structure_id=sid,canonical_mask=key) for sid,key in [("novel",novel),("novel",novel),("known","0011")]]
    rows.append(dict(rows[0],status="failed",scoring_structure_id="",canonical_mask=""))
    with attempts.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    schema_paths=[]
    for target, energies in [("deletion",[-3.,-2.,-1.,0.,1.]),("endpoint",[-4.,-3.,-2.,-1.,0.])]:
        schema=fit_schema([2]*5,energies,["train"]*5,n_sites=4,
            provenance=dict(endpoint_context_hash=context["context_hash"],energy_target=target))
        path=tmp_path/(target+".json");save_schema(schema,path);schema_paths.append(path)
    registered=tmp_path/"registered"
    original=tmp_path/"original_q20.csv"
    original.write_text("teacher,n_Li,source,q20_eV_atom\numa,2,main_train,-3.6\n")
    register.register(SimpleNamespace(prepared=prepared,pool=pool,attempts=attempts,label="synthetic",
        output_dir=registered,prior_task_files=[],reserved_other_protocol_attempts=0,maximum_primary_attempts=10,
        analysis_inputs=[*schema_paths,final_history,original]))
    registration=json.loads((registered/"registration.json").read_text())
    tasks=json.loads((registered/"tasks.json").read_text())["tasks"]
    queue=tmp_path/"queue";(queue/"claims").mkdir(parents=True);(queue/"results").mkdir()
    for task in tasks:
        directory=queue/"results"/task["task_id"];directory.mkdir()
        np.savez(directory/"arrays.npz",synthetic_only=[1])
        success=task["canonical_vacancy_key"]==novel
        common.atomic_json(directory/"result.json",dict(physical_key=task["physical_key"],context_hash=context["context_hash"],
            status="converged" if success else "step_limit",endpoint_labels_usable=success,
            initial_energy_eV=-14.,endpoint_energy_eV=-18.,n_atoms=4,n_target=2,
            mapping=dict(endpoint_occupation_orbit=task["canonical_vacancy_key"]),
            calculator_EFS_attempts=10,arrays_sha256=common.sha256(directory/"arrays.npz")))
        common.atomic_json(queue/"claims"/(task["task_id"]+".json"),dict(claimed_unix=registration["frozen_unix"]+(1 if success else -1)))
    args=SimpleNamespace(prepared=prepared,registration=registered,queue=queue,attempts=attempts,
        deletion_schema=schema_paths[0],endpoint_schema=schema_paths[1],history=final_history,
        output_dir=tmp_path/"evaluation",allow_incomplete=False,unseal_test=False,original_thresholds=original)
    receipt=evaluate.evaluate(args)
    assert receipt["status"]=="PASS_COMPLETE" and receipt["attempts"]==4
    assert receipt["unknown_endpoint_attempts"]==2
    assert receipt["newly_claimed_endpoint_tasks"]==receipt["cached_preexisting_compatible_tasks"]==1
    with (args.output_dir/"cells.csv").open() as stream:cell=next(csv.DictReader(stream))
    assert int(cell["attempts"])==4 and int(cell["endpoint_useful_novel_orbits"])==1
    assert float(cell["endpoint_useful_yield"])==.25
    assert float(cell["deletion_useful_yield"])==.25
    assert float(cell["original_q20_deletion_useful_yield"])==0
    with (args.output_dir/"fit_summary.csv").open() as stream:fit_rows=list(csv.DictReader(stream))
    assert len(fit_rows)==2 and all(float(r["equal_weight_count_request_yield"])==.25 for r in fit_rows)
    # Threshold changes after scoring never become a successful replay.
    final_history.write_text("[]")
    args.output_dir=tmp_path/"changed_history"
    with pytest.raises(ValueError,match="not frozen"):
        evaluate.evaluate(args)
