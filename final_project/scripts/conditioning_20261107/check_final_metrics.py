"""Independent raw-mask/scalar audit of the frozen FINAL assessment.

No evaluator helpers, model inference or new physical calculations are used.
Bin boundaries are exact frozen numbers, with exact boundary ties in the lower
bin. Only comparisons of independently computed floating-point aggregates allow
a small IEEE-754 arithmetic allowance; integer counts and labels must be exact.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np

from cathode_cvae.runtime import require_compute_allocation


ROOT=Path(__file__).resolve().parents[2]
GROUP=["teacher","method","replicate_id","fit_seed","sampling_seed","n_Li","requested_bin",
       "temperature_K","allocation_policy","request_schema_id","measured_schema_id","checkpoint_sha256"]
STRATUM=["teacher","method","temperature_K","allocation_policy","request_schema_id","measured_schema_id"]
OUTCOMES=["low","middle","high","failed_request","invalid_mask","unknown_energy"]
FLOAT_ULPS=8


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x",newline="") as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def bin_label(energy,edges):
    # Deliberately independent of the production searchsorted implementation.
    return "low" if energy<=edges[0] else "middle" if energy<=edges[1] else "high"


def mean(x):return float(sum(x)/len(x))


def quantile(values,q):
    if not values:return ""
    x=sorted(values);p=(len(x)-1)*q;lo=math.floor(p);hi=math.ceil(p)
    return float(x[lo]+(p-lo)*(x[hi]-x[lo]))


def canonical(x,permutations):
    """Exhaust all actual operations; numeric minima of padded vacancy codes."""
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=48 or not np.isin(x,[0,1]).all():
        raise ValueError("Actual benchmark masks must be binary48-site rows")
    best=np.full(len(x),2**64-1,np.uint64)
    for operation in permutations:
        transformed=1-x[:,np.argsort(operation)]
        packed=np.zeros((len(x),8),np.uint8)
        packed[:,:6]=np.packbits(transformed,axis=1,bitorder="big")
        value=packed.view(">u8").ravel().astype(np.uint64)
        best=np.minimum(best,value)
    return np.asarray([format(int(code)>>16,"048b") for code in best])


def schema(path):
    s=json.loads(Path(path).read_text())
    body={k:v for k,v in s.items() if k!="identity"}
    identity=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if identity!=s["identity"] or s["n_sites"]!=48 or s["units"]!="eV/atom":
        raise ValueError("Frozen teacher energy schema identity/units differ")
    for item in s["compositions"].values():
        if item["edges_eV_atom"]!=[item["q20_eV_atom"],item["q80_eV_atom"]]:
            raise ValueError("Frozen q20/q80 and classification boundaries disagree")
    return s


def cache_history(root,history_path):
    history=set(json.loads(history_path.read_text()));records=[];all_missing=set()
    for teacher in ["uma","mace"]:
        path=root/f"runs/conditioning_20261107/oracle/cache_{teacher}.npz"
        with np.load(path,allow_pickle=False) as z:ids=z["structure_ids"].astype(str)
        if len(ids)!=len(set(ids)):raise ValueError("Imported cache IDs repeat")
        unsupported=[];keys=[]
        for sid in ids:
            match=re.fullmatch(r"mask48_([0-9a-fA-F]{12})",sid)
            if not match:unsupported.append(sid)
            else:keys.append(format(int(match.group(1),16),"048b"))
        missing=sorted(set(keys)-history);all_missing.update(missing)
        records.append(dict(teacher=teacher,cache_path=str(path),cache_sha256=digest(path),cache_rows=len(ids),
            convertible_mask48_ids=len(keys),unsupported_ids=unsupported,missing_history_keys=missing,
            all_cache_orbits_known_before_final=not missing and not unsupported,energy_arrays_read=0))
    return dict(status="PASS" if all(r["all_cache_orbits_known_before_final"] for r in records) else "MISSING_KNOWN_HISTORY",
        frozen_history_rows=len(history),frozen_history_sha256=digest(history_path),teachers=records,
        distinct_missing_history_keys=sorted(all_missing),history_modified=False)


def candidates(path,parent_hash,perms,layer_ids,layer_sites):
    with np.load(path,allow_pickle=False) as z:
        ids=z["structure_ids"].astype(str);o=z["occupancy"];m=z["masks"]
        n=z["n_Li"];na=z["n_atoms"];claimed=str(z["parent_sha256"].item())
    if claimed!=parent_hash or len(set(ids))!=len(ids) or not np.array_equal(m,1-o):
        raise ValueError("Candidate geometry identity or occupancy/vacancy convention differs")
    keys=canonical(o,perms)
    result={}
    for sid,key,row,count,atoms in zip(ids,keys,o,n,na):
        expected_id=f"mask48_{int(key,2):012x}"
        if sid!=expected_id or int(count)!=int(row.sum()) or int(atoms)!=144+int(count):
            raise ValueError("Candidate ID/count/atom count does not encode its actual parent mask")
        # Geometry-coordinate diagnostic only, with the frozen six-decimal
        # fractional-layer convention; no physical phase is inferred.
        filled=np.bincount(layer_ids,weights=row,minlength=len(layer_sites))/layer_sites
        layer=float(np.sqrt(np.mean((filled-np.mean(filled))**2)))
        result[sid]=dict(key=key,n=int(count),layer=layer)
    return result


def scores(path,candidate):
    with np.load(path,allow_pickle=False) as z:
        ids=z["structure_ids"].astype(str);energy=z["energy_eV"].astype(float);atoms=z["n_atoms"]
        status=z["status"].astype(str) if "status" in z else np.full(len(ids),"success")
        provenance=z["provenance"].astype(str) if "provenance" in z else z["cached_source"].astype(str) if "cached_source" in z else np.full(len(ids),"unspecified")
        n=z["n_Li"] if "n_Li" in z else None
    if len(ids)!=len(set(ids)) or energy.shape!=ids.shape or atoms.shape!=ids.shape:
        raise ValueError("Final scalar IDs or array shapes are ambiguous")
    if status.ndim==0:status=np.repeat(status,len(ids))
    if provenance.ndim==0:provenance=np.repeat(provenance,len(ids))
    result={}
    for i,sid in enumerate(ids):
        if sid not in candidate:continue
        claimed=status[i].lower() in {"success","complete","completed","ok","fresh","cached"}
        valid=bool(claimed and np.isfinite(energy[i]) and np.isfinite(atoms[i]) and atoms[i]>0)
        if claimed and (int(atoms[i])!=atoms[i] or atoms[i]!=144+candidate[sid]["n"]):
            raise ValueError("Scalar result belongs to a different atom count")
        if n is not None and claimed and int(n[i])!=candidate[sid]["n"]:
            raise ValueError("Scalar result belongs to a different composition")
        result[sid]=dict(known=valid,e=float(energy[i]/atoms[i]) if valid else None,provenance=str(provenance[i]))
    return result


def metadata(row,teacher,identity):
    return dict(teacher=teacher,method=row["method"],replicate_id=row["replicate_id"],fit_seed=row["fit_seed"],
        sampling_seed=row["sampling_seed"],n_Li=int(row["n_Li"]),requested_bin=row["requested_bin"],
        temperature_K=row.get("temperature_K",""),allocation_policy=row["allocation_policy"],
        request_schema_id=row["condition_schema_id"],measured_schema_id=identity,
        checkpoint_sha256=row["checkpoint_sha256"])


def key(row,fields):return tuple(str(row[k]) for k in fields)


def fitted(row):
    absent={"","none","null","n/a","not-applicable"}
    return str(row["fit_seed"]).lower() not in absent and str(row["checkpoint_sha256"]).lower() not in absent


def observations(rows,teacher,condition,candidate,oracle,train,history,thresholds):
    out=[]
    for position,r in enumerate(rows):
        meta=metadata(r,teacher,condition["identity"]);sid=r["scoring_structure_id"]
        geometry=candidate.get(sid);generated=r["status"]=="generated"
        valid=bool(generated and geometry is not None);exact=bool(valid and geometry["n"]==meta["n_Li"])
        if valid and r["canonical_mask"]!=geometry["key"]:raise ValueError("Attempt and candidate name different mask orbits")
        scalar=oracle.get(sid) if exact else None;known=bool(scalar and scalar["known"])
        if not generated:label="failed_request"
        elif not exact:label="invalid_mask"
        elif not known:label="unknown_energy"
        else:label=bin_label(scalar["e"],condition["compositions"][str(meta["n_Li"])]["edges_eV_atom"])
        orbit=geometry["key"] if exact else "";energy=scalar["e"] if known else None
        q=thresholds.get((teacher,meta["n_Li"]))
        out.append(dict(meta,attempt_id=r["attempt_id"],source_position=position,scoring_structure_id=sid,
            canonical_mask=orbit,generated=generated,mask_valid=valid,exact_count=exact,energy_known=known,
            measured_bin=label,in_requested_bin=bool(known and label==meta["requested_bin"]),
            novel_training=bool(exact and orbit not in train),novel_history=bool(exact and orbit not in history),
            novel_both=bool(exact and orbit not in train and orbit not in history),energy_eV_atom=energy if known else "",
            fixed_q20_eV_atom=q if q is not None else "",passes_fixed_q20=bool(known and q is not None and energy<=q),
            score_provenance=scalar["provenance"] if known else "unknown",layer_imbalance_std_fraction=geometry["layer"] if exact else ""))
    return out


def measure(observed):
    grouped=defaultdict(list)
    for r in observed:grouped[key(r,GROUP)].append(r)
    metrics=[];confusion=[];curves=[]
    for _,rr in sorted(grouped.items()):
        base={k:rr[0][k] for k in GROUP};den=len(rr)
        unique={r["canonical_mask"] for r in rr if r["exact_count"]}
        exact=sum(r["exact_count"] for r in rr);known=sum(r["energy_known"] for r in rr)
        train={r["canonical_mask"] for r in rr if r["novel_training"]}
        history={r["canonical_mask"] for r in rr if r["novel_history"]}
        both={r["canonical_mask"] for r in rr if r["novel_both"]}
        useful={r["canonical_mask"] for r in rr if r["novel_both"] and r["in_requested_bin"]}
        low={r["canonical_mask"] for r in rr if r["novel_both"] and r["passes_fixed_q20"]}
        e=[r["energy_eV_atom"] for r in rr if r["energy_known"]]
        counts=Counter(r["measured_bin"] for r in rr);q_available=all(r["fixed_q20_eV_atom"]!="" for r in rr)
        layer=[r["layer_imbalance_std_fraction"] for r in rr if r["layer_imbalance_std_fraction"]!=""]
        hits=sum(r["in_requested_bin"] for r in rr)
        metrics.append(dict(base,attempts=den,generated=sum(r["generated"] for r in rr),
            request_failures=counts["failed_request"],invalid_masks=counts["invalid_mask"],
            mask_validity_rate=sum(r["mask_valid"] for r in rr)/den,exact_count_rate=exact/den,
            scored_attempts=known,score_coverage_per_attempt=known/den,
            distinct_scored_geometries=len({r["scoring_structure_id"] for r in rr if r["energy_known"]}),
            unique_orbits=len(unique),unique_orbit_yield_per_attempt=len(unique)/den,
            orbit_uniqueness_among_exact_count=len(unique)/exact if exact else "",
            duplicate_fraction_among_exact_count=1-len(unique)/exact if exact else "",
            distinct_novel_training_orbits=len(train),distinct_novel_history_orbits=len(history),distinct_novel_both_orbits=len(both),
            novel_training_yield_per_attempt=len(train)/den,novel_history_yield_per_attempt=len(history)/den,
            training_novel_fraction_of_unique=len(train)/len(unique) if unique else "",
            history_novel_fraction_of_unique=len(history)/len(unique) if unique else "",
            bin_hits=hits,bin_hit_rate_per_attempt=hits/den,bin_hit_rate_scored_valid=hits/known if known else "",
            distinct_novel_inbin_orbits=len(useful),novel_inbin_yield_per_attempt=len(useful)/den,
            historical_q20_available=q_available,distinct_novel_fixed_q20_orbits=len(low) if q_available else "",
            novel_fixed_q20_yield_per_attempt=len(low)/den if q_available else "",
            best_observed_energy_eV_atom=min(e) if e else "",energy_p10_eV_atom=quantile(e,.1),
            energy_median_eV_atom=quantile(e,.5),energy_p90_eV_atom=quantile(e,.9),
            mean_layer_imbalance_std_fraction=mean(layer) if layer else ""))
        confusion.extend(dict(base,measured_outcome=label,count=counts[label],fraction_all_attempts=counts[label]/den) for label in OUTCOMES)
        query=set();fresh=set();useful_seen=set();low_seen=set();novel=set();best=None;scored=0
        for i,r in enumerate(rr,1):
            if r["exact_count"]:query.add(r["scoring_structure_id"])
            if r["novel_both"]:novel.add(r["canonical_mask"])
            if r["energy_known"]:
                scored+=1;best=r["energy_eV_atom"] if best is None else min(best,r["energy_eV_atom"])
                if r["score_provenance"].lower() in {"fresh","new"}:fresh.add(r["scoring_structure_id"])
                if r["novel_both"] and r["in_requested_bin"]:useful_seen.add(r["canonical_mask"])
                if r["novel_both"] and r["passes_fixed_q20"]:low_seen.add(r["canonical_mask"])
            curves.append(dict(base,proposal_attempts=i,unique_score_queries_seen=len(query),
                unique_fresh_labels_encountered=len(fresh),score_coverage_per_attempt=scored/i,
                distinct_novel_orbits_seen=len(novel),distinct_novel_inbin_orbits_seen=len(useful_seen),
                distinct_novel_fixed_q20_orbits_seen=len(low_seen),best_observed_energy_eV_atom=best if best is not None else ""))
    return metrics,confusion,curves


def aggregate(metrics):
    strata=defaultdict(list)
    for r in metrics:strata[key(r,STRATUM)].append(r)
    summaries=[];seeds=[];requests=[]
    for _,rows in sorted(strata.items()):
        base={k:rows[0][k] for k in STRATUM};replicates=defaultdict(list)
        for r in rows:replicates[r["replicate_id"]].append(r)
        values=[]
        for seed,rr in sorted(replicates.items()):
            cells=defaultdict(list)
            for r in rr:cells[(r["n_Li"],r["requested_bin"])].append(r)
            value=mean([mean([r["novel_inbin_yield_per_attempt"] for r in cc]) for cc in cells.values()]);values.append(value)
            lows=[mean([r["novel_fixed_q20_yield_per_attempt"] for r in cc if r["novel_fixed_q20_yield_per_attempt"]!=""]) for cc in cells.values()]
            seeds.append(dict(base,replicate_id=seed,fit_seeds="|".join(sorted({r["fit_seed"] for r in rr})),
                condition_cells=len(cells),sampling_seeds="|".join(sorted({r["sampling_seed"] for r in rr})),
                macro_composition_bin_inbin_yield=value,macro_fixed_q20_yield_all_requested_bins=mean(lows),
                pooled_inbin_yield=sum(r["distinct_novel_inbin_orbits"] for r in rr)/sum(r["attempts"] for r in rr),
                has_fitted_checkpoint=any(fitted(r) for r in rr)))
        summaries.append(dict(base,replicate_count=len(replicates),design_seed_count=len(replicates),
            fitted_seed_count=len({r["fit_seed"] for r in rows if fitted(r)}),sampling_seed_count=len({r["sampling_seed"] for r in rows}),
            mean_novel_inbin_yield=mean(values),minimum_design_seed_yield=min(values),maximum_design_seed_yield=max(values),
            design_seed_standard_deviation=math.sqrt(sum((v-mean(values))**2 for v in values)/(len(values)-1)) if len(values)>1 else "",
            condition_grids=";".join(f"{s}:"+"|".join(f"{n}:{b}" for n,b in sorted({(r['n_Li'],r['requested_bin']) for r in rr})) for s,rr in sorted(replicates.items()))))
        for requested in sorted({r["requested_bin"] for r in rows}):
            per_seed=[]
            for seed,rr in sorted(replicates.items()):
                bycount=defaultdict(list)
                for r in rr:
                    if r["requested_bin"]==requested:bycount[r["n_Li"]].append(r)
                per_seed.append(dict(inbin=mean([mean([r["novel_inbin_yield_per_attempt"] for r in cc]) for cc in bycount.values()]),
                    low=mean([mean([r["novel_fixed_q20_yield_per_attempt"] for r in cc]) for cc in bycount.values()]),
                    hit=mean([mean([r["bin_hit_rate_per_attempt"] for r in cc]) for cc in bycount.values()])))
            requests.append(dict(base,requested_bin=requested,design_seed_count=len(per_seed),
                mean_novel_inbin_yield=mean([r["inbin"] for r in per_seed]),
                mean_novel_fixed_q20_yield=mean([r["low"] for r in per_seed]),mean_bin_hit_rate_per_attempt=mean([r["hit"] for r in per_seed])))
    return summaries,seeds,requests


def compare(path,expected,key_fields):
    archived=read(path);bykey={key(r,key_fields):r for r in archived}
    if len(bykey)!=len(archived) or set(bykey)!={key(r,key_fields) for r in expected}:
        raise AssertionError(f"Table identity/cardinality differs: {path}")
    differences=[];checks=0;largest=0.
    for row in expected:
        identity=key(row,key_fields);actual=bykey[identity]
        for field,want in row.items():
            checks+=1;got=actual.get(field)
            if isinstance(want,(bool,np.bool_)):equal=got==str(bool(want))
            elif isinstance(want,(int,np.integer)):
                try:equal=int(got)==want and float(got)==want
                except (TypeError,ValueError):equal=False
            elif isinstance(want,(float,np.floating)):
                try:
                    value=float(got);error=abs(value-want);largest=max(largest,error)
                    equal=math.isfinite(value) and error<=FLOAT_ULPS*np.finfo(float).eps*max(1.,abs(value),abs(want))
                except (TypeError,ValueError):equal=False
            else:equal=got==str(want)
            if not equal:differences.append(dict(identity=identity,field=field,expected=want,archived=got))
    return dict(path=str(path),rows=len(expected),field_checks=checks,maximum_absolute_float_difference=largest,
        mismatch_count=len(differences),first_mismatches=differences[:50],sha256=digest(path))


def selfcheck():
    assert [bin_label(e,[1.,2.]) for e in [0.,1.,1.5,2.,3.]]==["low","low","middle","middle","high"]
    # A hardcoded rotating 4-site pattern, embedded in a48-site identity host,
    # tests all operations without production canonicalization helpers.
    perms=np.tile(np.arange(48),(4,1))
    for j in range(4):perms[j,:4]=np.roll(np.arange(4),j)
    x=np.zeros((2,48),np.uint8);x[0,[0,1]]=1;x[1,[1,2]]=1
    assert canonical(x,perms)[0]==canonical(x,perms)[1]
    assert quantile([1.,2.,3.,4.],.1)==1.3
    return dict(status="PASS",exact_tie_cases=5,symmetry_related_mask_pair=True,linear_quantile_check=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output-dir",type=Path)
    p.add_argument("--history-only",action="store_true");p.add_argument("--self-test",action="store_true")
    a=p.parse_args(argv);require_compute_allocation()
    if a.self_test:print(json.dumps(selfcheck(),indent=2));return
    start=time.monotonic();root=a.root.resolve();run=root/"runs/conditioning_20261107"
    output=a.output_dir or root/("reports/conditioning_20261107/final_cache_history_check" if a.history_only else "reports/conditioning_20261107/independent_final_metrics")
    output.mkdir(parents=True,exist_ok=False)
    inputs={};audits=[]
    try:
        check=selfcheck();protocol=json.loads((run/"final_protocol.json").read_text())
        history_path=run/"data/history_before_final.json"
        if digest(history_path)!=protocol["history_sha256"]:raise ValueError("Pre-final frozen history changed")
        cache_check=cache_history(root,history_path)
        with (output/"cache_history.json").open("x") as f:json.dump(cache_check,f,indent=2);f.write("\n")
        if a.history_only:
            print(json.dumps(cache_check,indent=2))
            if cache_check["status"]!="PASS":raise SystemExit(1)
            return
        # The integrating owner launches this branch only after both final
        # evaluation receipts are complete. Check that boundary before opening
        # any final scalar score archive.
        for stage in ["final","final_matched"]:
            path=run/f"evaluation/{stage}/receipt.json"
            if json.loads(path.read_text())["status"]!="PASS":raise ValueError("Await complete integrating-owner final evaluation")
        parent_path=root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"
        data_path=root/"artifacts/current/improvement_v1/training_round0.npz"
        if digest(parent_path)!=protocol["parent_sha256"] or digest(data_path)!=protocol["data_sha256"]:
            raise ValueError("Protected parent or round-zero input changed")
        with np.load(parent_path,allow_pickle=False) as z:
            perms=z["permutations"];numbers=z["numbers"];pos=z["positions_A"];cell=z["cell_A"]
            mobile=z["target_indices"] if "target_indices" in z else z["li_indices"]
        if perms.shape!=(192,48) or not np.all(np.sort(perms,axis=1)==np.arange(48)) or len(numbers)!=192:
            raise ValueError("Unexpected parent symmetry/host identity")
        fractional=np.linalg.solve(cell.T,pos[mobile].T).T
        _,layers=np.unique(np.mod(np.round(np.mod(fractional[:,2],1),6),1),return_inverse=True)
        layer_sites=np.bincount(layers)
        with np.load(data_path,allow_pickle=False) as z:train=set(z["canonical_masks"][z["split"].astype(str)=="train"].astype(str))
        history=set(json.loads(history_path.read_text()))
        if len(train)!=40157 or len(history)!=278382 or not train<=history:raise ValueError("Protected training/history scope changed")
        training_keys=run/"data/training_keys.json"
        if set(json.loads(training_keys.read_text()))!=train:raise ValueError("Evaluator training keys differ from actual protected roles")
        full=run/"final";raw_path=full/"raw_attempts.npz";full_attempts=read(full/"attempts.csv")
        generated=json.loads((full/"receipt.json").read_text())
        if generated["manifest_sha256"]!=protocol["generation_manifest_sha256"]:raise ValueError("Final model/proposal manifest changed")
        for name,expected in generated["outputs"].items():
            if digest(full/name)!=expected:raise ValueError("Frozen final proposal bytes changed")
        with np.load(raw_path,allow_pickle=False) as z:raw=z["occupancy"]
        raw_keys=canonical(raw,perms);byattempt={r["attempt_id"]:r for r in full_attempts}
        if len(byattempt)!=34560 or raw.shape!=(34560,48):raise ValueError("Wrong final primary budget")
        for r in full_attempts:
            index=int(r["raw_row"])
            if raw_keys[index]!=r["canonical_mask"] or int(raw[index].sum())!=int(r["n_Li"]):
                raise ValueError("Final raw mask/canonical key/count does not match its attempt identity")
        matched=run/"final_matched_audit";subset=read(matched/"attempts.csv")
        subset_receipt=json.loads((matched/"receipt.json").read_text())
        expected_ids={r["attempt_id"] for r in full_attempts if r["method"] in protocol["MACE"]["families"]}
        if {r["attempt_id"] for r in subset}!=expected_ids or len(subset)!=20736:
            raise ValueError("MACE audit was not the predeclared method-only subset")
        for r in subset:
            if r!=byattempt[r["attempt_id"]]:raise ValueError("Matched audit changed an attempt beyond method-only selection")
        for name,expected in subset_receipt["outputs"].items():
            if digest(matched/name)!=expected:raise ValueError("Matched-audit bytes changed")
        schema_paths={t:root/f"artifacts/overnight_20261107/data/condition_schema_{t}.json" for t in ["uma","mace"]}
        conditions={t:schema(path) for t,path in schema_paths.items()}
        threshold_path=root/"artifacts/current/reports/generation_thresholds.csv"
        thresholds={(r["teacher"],int(r["n_Li"])):float(r["q20_eV_atom"]) for r in read(threshold_path) if r["source"]=="main_train"}
        all_sources=[run/"final_protocol.json",run/"final_generation_manifest.json",history_path,training_keys,parent_path,data_path,raw_path,
            full/"attempts.csv",full/"candidates.npz",full/"receipt.json",matched/"attempts.csv",matched/"candidates.npz",matched/"receipt.json",
            threshold_path,*schema_paths.values(),run/"collected/final/uma.npz",run/"collected/final/mace.npz"]
        inputs={str(q.relative_to(root)):digest(q) for q in all_sources}
        for stage,folder,rows,teachers in [("final",full,full_attempts,["uma"]),("final_matched",matched,subset,["uma","mace"])]:
            candidate=candidates(folder/"candidates.npz",protocol["parent_sha256"],perms,layers,layer_sites)
            if set(candidate)!={r["scoring_structure_id"] for r in rows}:raise ValueError("Candidate pool does not match complete attempted geometry set")
            records=[]
            for teacher in teachers:
                scalar=scores(run/f"collected/final/{teacher}.npz",candidate)
                records.extend(observations(rows,teacher,conditions[teacher],candidate,scalar,train,history,thresholds))
            if any(not r["energy_known"] for r in records):raise ValueError("Final oracle coverage is not complete")
            metrics,confusion,curves=measure(records);summary,seeds,requests=aggregate(metrics)
            budgets=Counter((r["teacher"],r["method"],r["replicate_id"],r["n_Li"],r["requested_bin"]) for r in records)
            if set(budgets.values())!={128}:raise ValueError("A final condition/seed cell has the wrong attempt denominator")
            tables=[("metrics.csv",metrics,GROUP),("confusion.csv",confusion,GROUP+["measured_outcome"]),
                ("seed_metrics.csv",seeds,STRATUM+["replicate_id"]),("summary.csv",summary,STRATUM),
                ("summary_by_request.csv",requests,STRATUM+["requested_bin"]),
                ("fixed_q20_low_request.csv",[r for r in requests if r["requested_bin"]=="low"],STRATUM+["requested_bin"]),
                ("budget_curves.csv",curves,GROUP+["proposal_attempts"])]
            for name,expected,identity in tables:
                audit=compare(run/f"evaluation/{stage}/{name}",expected,identity);audits.append(audit)
                if name!="budget_curves.csv":write(output/f"{stage}_independent_{name}",expected)
            # An additional attempt-level check confirms the independently
            # reconstructed bins/novelty/scalars, rather than trusting the
            # production attempt-outcome join as input.
            fields=["teacher","attempt_id","canonical_mask","exact_count","energy_known","measured_bin",
                "novel_training","novel_history","novel_both","energy_eV_atom","passes_fixed_q20","in_requested_bin"]
            audits.append(compare(run/f"evaluation/{stage}/attempt_outcomes.csv",
                [{k:r[k] for k in fields} for r in records],["teacher","attempt_id"]))
        after={str(q.relative_to(root)):digest(q) for q in all_sources}
        if after!=inputs:raise ValueError("Raw final evidence changed during independent audit")
        mismatches=sum(r["mismatch_count"] for r in audits)
        result=dict(status="PASS" if not mismatches and cache_check["status"]=="PASS" else "FAILED_MISMATCHES",
            elapsed_seconds=time.monotonic()-start,raw_final_attempts=34560,matched_attempts=20736,
            teachers="UMA full final; UMA/MACE exact predeclared matched subset",parent_operations=192,
            protected_train_orbits=len(train),frozen_history_orbits=len(history),
            comparisons=audits,field_checks=sum(r["field_checks"] for r in audits),mismatch_count=mismatches,
            selfcheck=check,cache_history_status=cache_check["status"],input_hashes=inputs,inputs_unchanged=True,
            float_comparison=dict(multiplier=FLOAT_ULPS,epsilon=float(np.finfo(float).eps),
                formula="abs(a-b)<=8*eps64*max(1,abs(a),abs(b)) only for arithmetic aggregates; integer counts, canonical identities, labels and boundary comparisons exact",
                interpretation="Floating-point arithmetic allowance, never a scientific success threshold"),
            source_sha256=digest(__file__),production_evaluator_helpers_imported=False,new_MLIP_calls=0,new_training_trajectories=0,
            final_used_for_policy_selection=False,job_id=os.environ.get("SLURM_JOB_ID"),step_id=os.environ.get("SLURM_STEP_ID"))
        with (output/"receipt.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
        print(json.dumps(result,indent=2))
        if result["status"]!="PASS":raise SystemExit(1)
    except Exception as error:
        with (output/"failure_receipt.json").open("x") as f:json.dump(dict(status="FAILED_PRESERVED",error=str(error),
            traceback=traceback.format_exc(),source_sha256=digest(__file__),input_hashes=inputs,
            elapsed_seconds=time.monotonic()-start,new_MLIP_calls=0),f,indent=2)
        raise


if __name__=="__main__":main()
