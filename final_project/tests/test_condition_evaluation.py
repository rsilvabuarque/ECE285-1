"""Synthetic controlled cases for attempt accounting and teacher-specific bins."""

import csv
import json

import numpy as np
import pytest

from cathode_cvae.condition_evaluation import evaluate
from cathode_cvae.conditions import fit_schema, save_schema, schema_identity


def write(path, rows):
    with path.open("w", newline="") as f:
        names = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=names); w.writeheader(); w.writerows(rows)


def read(path):
    return list(csv.DictReader(path.open()))


def setup_case(tmp_path):
    x = np.zeros((4, 48), np.uint8)
    for i in range(4): x[i, np.arange(i, i+12)] = 1
    keys = ["".join(map(str, (1-row).tolist())) for row in x]
    candidates = tmp_path/"candidates.npz"
    np.savez(candidates, occupancy=x, canonical_masks=keys, structure_ids=["a","b","c","d"])
    schemas, scores = {}, {}
    for teacher, shift in [("uma",0.),("mace",100.)]:
        path = tmp_path/f"{teacher}_schema.json"
        schema = fit_schema(np.full(10,12), np.array([0,0,0,1,1,1,1,2,3,4])+shift,
                            ["train"]*10, n_sites=48)
        save_schema(schema,path);schemas[teacher]=path
        path = tmp_path/f"{teacher}_scores.npz"
        np.savez(path, structure_ids=["a","b","c","d"],
                 energy_eV=(np.array([-1.,-.9,3.,np.nan])+shift)*156,
                 n_atoms=[156,156,156,np.nan], n_Li=[12]*4,
                 status=["success","success","success","failed"],provenance=["fresh"]*4)
        scores[teacher]=path
    rows=[]
    for i,sid in enumerate(["a","b","b","c","d",""]):
        rows.append(dict(attempt_id=f"attempt{i}",method="multibin",replicate_id="r1",fit_seed="1",sampling_seed="3",
            n_Li=12, requested_bin="low",status="generated" if sid else "failed_request",
            scoring_structure_id=sid,canonical_mask=keys[["a","b","c","d"].index(sid)] if sid else "",
            condition_schema_id=schema_identity(json.loads(schemas["uma"].read_text())),
            temperature_K="",allocation_policy="fixed-request",checkpoint_sha256="synthetic-checkpoint"))
    attempts=tmp_path/"attempts.csv";write(attempts,rows)
    training=tmp_path/"training.json";training.write_text(json.dumps([keys[0]]))
    history=tmp_path/"history.json";history.write_text(json.dumps([keys[0]]))
    threshold=tmp_path/"threshold.csv"
    write(threshold,[dict(teacher=t,n_Li=12,source="main_train",q20_eV_atom=-.5+s) for t,s in [("uma",0), ("mace",100)]])
    return dict(attempts_path=attempts,candidates_path=candidates,score_paths=scores,schema_paths=schemas,
                training_path=training,history_path=history,historical_thresholds=threshold), keys


def test_all_attempts_remain_and_each_teacher_uses_its_own_energy_partition(tmp_path):
    kwargs, keys=setup_case(tmp_path);out=tmp_path/"out"
    receipt=evaluate(**kwargs,output_dir=out)
    assert receipt["status"]=="INCOMPLETE_SCORE_COVERAGE"
    groups={r["teacher"]:r for r in read(out/"metrics.csv")}
    for row in groups.values():
        assert int(row["attempts"])==6
        assert int(row["unique_orbits"])==4
        assert int(row["distinct_novel_inbin_orbits"])==1
        assert float(row["novel_inbin_yield_per_attempt"])==pytest.approx(1/6)
        assert float(row["bin_hit_rate_per_attempt"])==.5
        assert float(row["bin_hit_rate_scored_valid"])==.75
        assert int(row["request_failures"])==1
        assert float(row["score_coverage_per_attempt"])==pytest.approx(4/6)
        assert int(row["distinct_novel_fixed_q20_orbits"])==1
    assert float(groups["mace"]["best_observed_energy_eV_atom"])-float(groups["uma"]["best_observed_energy_eV_atom"])==pytest.approx(100)
    conf=[r for r in read(out/"confusion.csv") if r["teacher"]=="uma"]
    counts={r["measured_outcome"]:int(r["count"]) for r in conf}
    assert counts==dict(low=3,middle=0,high=1,failed_request=1,invalid_mask=0,unknown_energy=1)
    curves=[r for r in read(out/"budget_curves.csv") if r["teacher"]=="uma"]
    assert int(curves[-1]["proposal_attempts"])==6
    assert int(curves[-1]["unique_score_queries_seen"])==4
    assert int(curves[-1]["distinct_novel_inbin_orbits_seen"])==1


def test_training_and_history_novelty_are_distinct_and_both_required(tmp_path):
    kwargs,keys=setup_case(tmp_path)
    kwargs["training_path"].write_text(json.dumps(keys[:2]))
    out=tmp_path/"out";evaluate(**kwargs,output_dir=out)
    row=next(r for r in read(out/"metrics.csv") if r["teacher"]=="uma")
    assert int(row["distinct_novel_training_orbits"])==2
    assert int(row["distinct_novel_history_orbits"])==3
    assert int(row["distinct_novel_inbin_orbits"])==0


def test_baseline_rng_replicates_are_not_counted_as_model_fits(tmp_path):
    kwargs,keys=setup_case(tmp_path)
    template=read(kwargs["attempts_path"])[1]
    rows=[]
    for i in range(3):
        rows.append(dict(template,attempt_id=f"base{i}",method="uniform",replicate_id=str(i),
                         fit_seed="",sampling_seed=str(100+i),checkpoint_sha256="not-applicable"))
    write(kwargs["attempts_path"],rows)
    out=tmp_path/"out";evaluate(**kwargs,output_dir=out)
    summaries=read(out/"summary.csv")
    assert len(summaries)==2
    for row in summaries:
        assert int(row["replicate_count"])==3
        assert int(row["fitted_seed_count"])==0
        assert int(row["sampling_seed_count"])==3


def test_mismatched_candidate_identity_is_rejected(tmp_path):
    kwargs,keys=setup_case(tmp_path)
    rows=read(kwargs["attempts_path"]);rows[0]["canonical_mask"]=keys[1];write(kwargs["attempts_path"],rows)
    with pytest.raises(ValueError,match="identity mismatch"):
        evaluate(**kwargs,output_dir=tmp_path/"out")


def test_historical_fixed_q20_comparison_uses_only_low_requests(tmp_path):
    kwargs,keys=setup_case(tmp_path)
    rows=read(kwargs["attempts_path"])
    high=dict(rows[3],attempt_id="high-attempt",sampling_seed="4",requested_bin="high")
    write(kwargs["attempts_path"],rows+[high])
    out=tmp_path/"out";evaluate(**kwargs,output_dir=out)
    low=next(r for r in read(out/"fixed_q20_low_request.csv") if r["teacher"]=="uma")
    seed=next(r for r in read(out/"seed_metrics.csv") if r["teacher"]=="uma")
    assert float(low["mean_novel_fixed_q20_yield"])==pytest.approx(1/6)
    assert float(seed["macro_fixed_q20_yield_all_requested_bins"])==pytest.approx(1/12)
