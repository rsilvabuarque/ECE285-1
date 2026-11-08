"""Independent frozen aggregate-Gaussian proposal audit; no neural inference.

Recomputes actual mask identities, coupled draws, transformed latent values and
UMA/MACE useful yields. Moment checks independently verify TRAIN membership and
analytic matrix identities, with explicit limits: per-row encoder outputs were
not saved, so this does not independently rerun their population moments.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,math,time
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()
def js(path):return json.loads(Path(path).read_text())
def csvrows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def npz(path,names=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (names or z.files)}
def write(path,rows):
    with Path(path).open("x",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def label(value,edges):return "low" if value<=edges[0] else "middle" if value<=edges[1] else "high"
def canonical(x,permutations):
    if permutations.shape!=(192,48) or not np.all(np.sort(permutations,axis=1)==np.arange(48)):raise ValueError("Require all192 parent actions")
    inverse=np.argsort(permutations,axis=1);powers=np.left_shift(np.uint64(1),np.arange(47,-1,-1,dtype=np.uint64));result=[]
    for start in range(0,len(x),128):
        codes=np.sum((1-x[start:start+128])[:,inverse].astype(np.uint64)*powers,axis=2).min(1)
        result.extend(format(int(c),"048b") for c in codes)
    return np.asarray(result)
def compare(path,computed,keys,fields):
    saved={tuple(str(r[k]) for k in keys):r for r in csvrows(path)}
    if len(saved)!=len(computed):raise ValueError(f"Group count differs:{path}")
    for row in computed:
        other=saved[tuple(str(row[k]) for k in keys)]
        for field in fields:
            if isinstance(row[field],int) and str(row[field])!=other[field]:raise ValueError(f"Exact count differs:{field}")
            if not math.isclose(float(row[field]),float(other[field]),abs_tol=1e-12,rel_tol=1e-10):raise ValueError(f"Metric differs:{field}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation();start=time.monotonic();root=a.root.resolve();run=root/"runs/conditioning_20261107"
    gen=run/"aggregate_prior/assessment";prep=run/"aggregate_prior/preparation";evaluation=run/"evaluation/aggregate_prior"
    protocol=js(gen/"protocol.json");receipt=js(gen/"receipt.json");preparation=js(prep/"receipt.json");prepared_protocol=js(prep/"protocol.json")
    if receipt["status"]!="FROZEN_BEFORE_SCORING" or receipt["protocol_sha256"]!=sha(gen/"protocol.json"):raise ValueError("Unfrozen proposals")
    for name,value in receipt["outputs"].items():
        if sha(gen/name)!=value:raise ValueError("Frozen proposal bytes changed")
    if protocol["moment_preparation_protocol_sha256"]!=sha(prep/"protocol.json") or preparation["protocol_sha256"]!=sha(prep/"protocol.json"):raise ValueError("Moment protocol changed")
    if preparation["status"]!="PASS_TRAIN_ONLY_MOMENTS" or sha(prep/"moments.npz")!=protocol["moments_sha256"]:raise ValueError("Unverified moment bytes")
    if sha(root/"scripts/conditioning_20261107/aggregate_prior_ablation.py")!=preparation["source_sha256"]:raise ValueError("Executed moment source differs")
    for binding in protocol["inputs"].values():
        if sha(binding["path"])!=binding["sha256"]:raise ValueError("Protected moment input changed")
    for checkpoint in protocol["checkpoints"]:
        if sha(checkpoint["path"])!=checkpoint["sha256"]:raise ValueError("Weights changed")
    parent=npz(protocol["inputs"]["parent"]["path"],["permutations"]);perms=parent["permutations"]
    data=npz(protocol["inputs"]["data"]["path"],["split","canonical_masks","n_Li","energy_eV_atom","structure_ids"])
    schema=js(protocol["inputs"]["schema"]["path"]);train_indices=np.flatnonzero(data["split"].astype(str)=="train")
    if len(train_indices)!=40157:raise ValueError("TRAIN row membership changed")
    train_labels=np.asarray([label(float(data["energy_eV_atom"][i]),schema["compositions"][str(int(data["n_Li"][i]))]["edges_eV_atom"]) for i in train_indices])
    moments=npz(prep/"moments.npz");records=js(prep/"moment_groups.json")
    moment_map={};moment_rows=[]
    expected={(s,n,b) for s in (20261106,20261107,20261108) for n in (12,24,36) for b in ("low","middle","high")}
    for i,record in enumerate(records):
        key=(int(record["fit_seed"]),int(record["n_Li"]),record["energy_bin"])
        if key in moment_map:raise ValueError("Moment group repeats")
        n,b=key[1:];selected=train_indices[(data["n_Li"][train_indices]==n)&(train_labels==b)]
        if len(selected)!=record["TRAIN_rows"] or len(selected)!=int(moments["TRAIN_rows"][i]):raise ValueError("Moments use wrong TRAIN class membership/count")
        if (int(moments["fit_seeds"][i]),int(moments["n_Li"][i]),str(moments["energy_bins"][i]))!=key:raise ValueError("Moment archive row mismatch")
        m,S,L=[np.asarray(record[name],float) for name in ("mean","covariance","cholesky")]
        for value,name in ((m,"means"),(S,"covariances"),(L,"cholesky")):
            if not np.array_equal(value,moments[name][i]):raise ValueError("Moment JSON and arrays disagree")
        residual=float(np.max(np.abs(L@L.T-S)));scale=max(1.,float(np.linalg.norm(S,ord=np.inf)))
        numerical_bound=64*np.finfo(float).eps*scale
        if residual>numerical_bound or np.linalg.eigvalsh(S).min()<=0:raise ValueError("Gaussian covariance/Cholesky identity fails")
        expected_variance=np.asarray(record["marginal_posterior_variance_mean"],float)
        if np.any(expected_variance<=0):raise ValueError("Posterior variance is not positive")
        between=S-np.diag(expected_variance);min_eig=float(np.linalg.eigvalsh(between).min())
        if min_eig< -numerical_bound:raise ValueError("Total covariance is inconsistent with positive population Cov(mu)")
        moment_map[key]=(m,L)
        moment_rows.append(dict(fit_seed=key[0],n_Li=n,energy_bin=b,TRAIN_rows=len(selected),
            TRAIN_source_rows_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
            TRAIN_structure_ids_sha256=hashlib.sha256("\n".join(data["structure_ids"][selected].astype(str)).encode()).hexdigest(),
            covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(S).min()),between_mean_covariance_minimum_eigenvalue=min_eig,
            cholesky_identity_maximum_absolute_residual=residual,arithmetic_consistency_bound=numerical_bound))
    if set(moment_map)!=expected or preparation["encoded_TRAIN_rows"]!=3*40157:raise ValueError("Missing27 TRAIN moment groups")
    raw=npz(gen/"raw_attempts.npz");rows=csvrows(gen/"attempts.csv");x=raw["occupancy"]
    if len(rows)!=6912 or x.shape!=(6912,48) or not np.isin(x,[0,1]).all():raise ValueError("Actual6912 raw proposals are invalid")
    keys=canonical(x,perms);candidate=npz(gen/"candidates.npz");candidate_keys=canonical(candidate["occupancy"],perms)
    if set(candidate_keys)!=set(keys) or len(set(candidate_keys))!=len(candidate_keys) or str(candidate["parent_sha256"].item())!=protocol["inputs"]["parent"]["sha256"]:
        raise ValueError("Complete scoring pool differs from the raw proposal orbits/parent")
    if any(sid!=f"mask48_{int(key,2):012x}" for sid,key in zip(candidate["structure_ids"].astype(str),candidate_keys)):raise ValueError("Candidate ID/canonical geometry mismatch")
    if any("".join(map(str,m.tolist()))!=key for m,key in zip(candidate["masks"],candidate_keys)):raise ValueError("Candidate geometry is not canonical")
    groups=defaultdict(list);common={};max_z_error=0.;max_roundoff_ratio=0.;shifted_groups=0
    for i,row in enumerate(rows):
        if int(row["raw_row"])!=i or row["status"]!="generated" or row["canonical_mask"]!=keys[i] or int(x[i].sum())!=int(row["n_Li"]):raise ValueError("Raw mask identity/count changed")
        group=(row["method"],int(row["fit_seed"]),int(row["n_Li"]),row["requested_bin"]);groups[group].append(i)
    wanted={(method,s,n,b) for method in ("standard_prior","aggregate_prior") for s,n,b in expected}
    if set(groups)!=wanted or any(len(indices)!=128 for indices in groups.values()):raise ValueError("Missing128-attempt group")
    for (method,fit,n,b),indices in groups.items():
        ii=sorted(indices,key=lambda i:int(rows[i]["attempt"]));eps=raw["base_epsilon"][ii];u=raw["uniforms"][ii];z=raw["latents"][ii]
        if eps.dtype!=np.float32 or u.dtype!=np.float32 or z.dtype!=np.float32 or not np.isfinite(eps).all() or not np.all((u>=0)&(u<1)):raise ValueError("Saved FP32 base draws are invalid")
        for array,field in ((eps,"base_epsilon_sha256"),(u,"uniforms_sha256"),(z,"latents_sha256")):
            value=hashlib.sha256(array.tobytes()).hexdigest()
            if any(rows[i][field]!=value for i in indices):raise ValueError("Actual draw fingerprint differs")
        if (fit,n) in common:
            e0,u0=common[fit,n]
            if not np.array_equal(eps,e0) or not np.array_equal(u,u0):raise ValueError("Baseepsilon or AR uniforms differ across methods/requests")
        else:common[fit,n]=(eps.copy(),u.copy())
        if method=="standard_prior":
            if not np.array_equal(z,eps):raise ValueError("Standard prior is not the common standard noise")
        else:
            m,L=moment_map[fit,n,b];m=m.astype(np.float32).astype(float);L=L.astype(np.float32).astype(float)
            reference=eps.astype(float)@L.T+m;error=np.abs(z.astype(float)-reference)
            # A4D dot plus center has at most9 scalar roundings. Compare against
            # a double-precision reference using the standard gamma9 bound.
            unit=np.finfo(np.float32).eps/2;gamma=9*unit/(1-9*unit)
            bound=gamma*(np.abs(eps.astype(float))@np.abs(L).T+np.abs(m))+np.finfo(float).tiny
            if np.any(error>bound):raise ValueError("Aggregate actual z differs beyond FP32 dot/add rounding")
            max_z_error=max(max_z_error,float(error.max()));max_roundoff_ratio=max(max_roundoff_ratio,float((error/bound).max()))
            shifted_groups+=int(not np.array_equal(z,eps))
    history=set(js(gen/"history.json"));training=set(data["canonical_masks"][train_indices].astype(str))
    if len(history)!=312715 or sha(gen/"history.json")!=protocol["history_sha256"] or not training<=history:raise ValueError("Frozen novelty boundary differs")
    score_dir=run/"collected/optional_final";collector=js(score_dir/"receipt.json")
    if collector["status"]!="PASS" or collector["registry_sha256"]!=sha(score_dir/"registry_snapshot.json"):raise ValueError("Complete labels lack immutable registry binding")
    metric=[];confusion=[];seed_stats=[];summary=[];low_summary=[]
    thresholds={(r["teacher"],int(r["n_Li"])):float(r["q20_eV_atom"]) for r in csvrows(root/"artifacts/current/reports/generation_thresholds.csv") if r["source"]=="main_train"}
    score_hashes={}
    for teacher in ("uma","mace"):
        scores=npz(score_dir/(teacher+".npz"),["structure_ids","energy_eV","n_atoms","n_Li"])
        ids=scores["structure_ids"].astype(str)
        if len(set(ids))!=len(ids):raise ValueError("Duplicate physical scalar IDs")
        lookup={sid:j for j,sid in enumerate(ids)};measured_schema=js(root/f"artifacts/overnight_20261107/data/condition_schema_{teacher}.json")
        score_hashes[teacher]=sha(score_dir/(teacher+".npz"))
        for (method,fit,n,b),indices in sorted(groups.items()):
            useful=set();fixed=set();hits=0;measured_bins=[]
            for i in indices:
                row=rows[i];sid=row["scoring_structure_id"]
                if sid!=f"mask48_{int(keys[i],2):012x}" or sid not in lookup:raise ValueError("Missing exact candidate score")
                j=lookup[sid]
                if int(scores["n_atoms"][j])!=144+n or int(scores["n_Li"][j])!=n or not np.isfinite(scores["energy_eV"][j]):raise ValueError("Score geometry/composition invalid")
                energy=float(scores["energy_eV"][j])/(144+n);actual=label(energy,measured_schema["compositions"][str(n)]["edges_eV_atom"])
                measured_bins.append(actual);hits+=actual==b;novel=keys[i] not in training and keys[i] not in history
                if novel and actual==b:useful.add(keys[i])
                if novel and energy<=thresholds[teacher,n]:fixed.add(keys[i])
            metric.append(dict(teacher=teacher,method=method,fit_seed=fit,n_Li=n,requested_bin=b,attempts=128,scored_attempts=128,
                distinct_novel_inbin_orbits=len(useful),novel_inbin_yield_per_attempt=len(useful)/128,
                distinct_novel_fixed_q20_orbits=len(fixed),novel_fixed_q20_yield_per_attempt=len(fixed)/128,bin_hit_rate_per_attempt=hits/128))
            for actual in ("low","middle","high"):
                confusion.append(dict(teacher=teacher,method=method,fit_seed=fit,n_Li=n,requested_bin=b,measured_bin=actual,count=measured_bins.count(actual)))
        for method in ("standard_prior","aggregate_prior"):
            for fit in (20261106,20261107,20261108):
                rr=[r for r in metric if r["teacher"]==teacher and r["method"]==method and r["fit_seed"]==fit]
                seed_stats.append(dict(teacher=teacher,method=method,replicate_id=fit,macro_composition_bin_inbin_yield=float(np.mean([r["novel_inbin_yield_per_attempt"] for r in rr])),
                    original_q20_LOW_request_yield=float(np.mean([r["novel_fixed_q20_yield_per_attempt"] for r in rr if r["requested_bin"]=="low"]))))
            ss=[r for r in seed_stats if r["teacher"]==teacher and r["method"]==method];v=[r["macro_composition_bin_inbin_yield"] for r in ss]
            summary.append(dict(teacher=teacher,method=method,replicate_count=3,fitted_seed_count=3,mean_novel_inbin_yield=float(np.mean(v)),design_seed_standard_deviation=float(np.std(v,ddof=1))))
            low_summary.append(dict(teacher=teacher,method=method,mean_novel_fixed_q20_yield=float(np.mean([r["original_q20_LOW_request_yield"] for r in ss]))))
    compare(evaluation/"metrics.csv",metric,["teacher","method","fit_seed","n_Li","requested_bin"],
        ["attempts","scored_attempts","distinct_novel_inbin_orbits","novel_inbin_yield_per_attempt","distinct_novel_fixed_q20_orbits","novel_fixed_q20_yield_per_attempt","bin_hit_rate_per_attempt"])
    compare(evaluation/"seed_metrics.csv",seed_stats,["teacher","method","replicate_id"],["macro_composition_bin_inbin_yield"])
    compare(evaluation/"summary.csv",summary,["teacher","method"],["replicate_count","fitted_seed_count","mean_novel_inbin_yield","design_seed_standard_deviation"])
    compare(evaluation/"fixed_q20_low_request.csv",low_summary,["teacher","method"],["mean_novel_fixed_q20_yield"])
    out=a.output_dir
    if out.exists():raise FileExistsError("Preserve the prior independent audit")
    out.mkdir(parents=True,exist_ok=False)
    for name,content in (("metrics.csv",metric),("confusion.csv",confusion),("seed_metrics.csv",seed_stats),("summary.csv",summary),
                         ("original_q20_LOW_request.csv",low_summary),("moment_membership_and_matrix_checks.csv",moment_rows)):
        write(out/name,content)
    result=dict(status="PASS",audited_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        attempts_per_teacher=6912,teachers=["uma","mace"],canonical_parent_operations=192,history_orbits=len(history),
        common_base_epsilon_and_AR_uniforms_verified=True,actual_z_checked_not_assumed_common=True,changed_aggregate_z_groups=shifted_groups,
        maximum_aggregate_z_transform_absolute_error=max_z_error,maximum_fraction_of_derived_FP32_rounding_bound=max_roundoff_ratio,
        moment_groups=27,TRAIN_rows_per_fit=40157,TRAIN_only_membership_counts_and_row_hashes_verified=True,
        moment_law="Population E[diag(exp(logvar))]+Cov(mu), denominator N, as audited in the hash-bound executed source; covariance/Cholesky/positive-between-mean identities checked independently",
        moment_limit="Per-row mu/logvar not saved; this audit does not re-encode masks or independently recompute each population moment from encoder outputs",
        model_reselection=False,new_NN_optimizer_updates=0,new_encoder_calls=0,new_MLIP_calls=0,
        independent_production_table_comparisons_passed=True,summary=summary,original_q20_LOW_request=low_summary,
        source_sha256=sha(__file__),score_sha256=score_hashes,registry_snapshot_sha256=sha(score_dir/"registry_snapshot.json"),
        source_bindings={str(path):sha(path) for path in [gen/"protocol.json",gen/"receipt.json",gen/"raw_attempts.npz",gen/"attempts.csv",gen/"history.json",prep/"moments.npz",prep/"moment_groups.json",prep/"receipt.json"]},
        interpretation="Posthoc class-dependent Gaussian proposals, not a learned conditional prior, exact posterior mixture, same-z condition-only test, or DFT validation")
    with (out/"receipt.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
