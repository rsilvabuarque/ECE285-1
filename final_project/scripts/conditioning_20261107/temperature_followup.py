"""Small, paired temperature-allocation follow-up after the required report gate.

Temperature changes only outer energy-bin requests. Existing total-cell UMA
energies and raw draw frequencies define finite-reference weights. No MD,
relaxation, DFT, model fitting, new reference pool, or MACE call is performed.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from cathode_cvae.acquisition import (KB_EV_K, allocate_integer, bin_probabilities,
    exploration_policy, load_reference, static_weights, validate_exhaustive_support)
from cathode_cvae.conditions import classify, load_schema, supported_bins
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation

COUNTS = (12, 24, 36)
TEMPERATURES = (393.15, 433.15, 473.15, 493.15)
BINS = ("low", "middle", "high")
BUDGET = 32
EXPLORATION = 0.1
NAMESPACE = "conditioning-temperature-followup-20261107-v1"


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def seed_for(*values):
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:4], "little")


def verify_sources(protocol):
    for name, source in protocol["sources"].items():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Frozen source changed: "+name)


def prepare(root, output, editor_pool):
    require_compute_allocation()
    gate = root/"artifacts/conditioning_20261107/required_gate.json"
    assert json.loads(gate.read_text())["status"] == "COMPLETE"
    output.mkdir(parents=True, exist_ok=False)
    parent = root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"
    reference = root/"artifacts/overnight_20261107/data/reference.npz"
    schema_path = root/"artifacts/overnight_20261107/data/condition_schema_uma.json"
    old_receipt = root/"reports/overnight_20261107/reference_diagnostics/receipt.json"
    selection = root/"artifacts/conditioning_20261107/architecture_selection.json"
    final_protocol = root/"runs/conditioning_20261107/final_protocol.json"
    final_pool = root/"runs/conditioning_20261107/final/candidates.npz"
    finite_parent = root/"artifacts/current/extras/learning_all_parents/parents/Hu_O3_12.npz"
    finite_data = root/"artifacts/current/extras/learning_all_parents/learning_Hu_O3_12_uma.npz"
    prior = json.loads(old_receipt.read_text())
    assert prior["status"] == "PASS"
    assert sha256(reference) == prior["source_hashes"]["joined_reference"]
    assert sha256(parent) == prior["parent_sha256"]
    ref = load_reference(reference, parent, "uniform_raw_sample")
    assert len(ref["occupancy"]) == 24575 and int(ref["measure"].sum()) == 24576
    d = ref["data"]
    assert str(d["teacher"].item()) == "uma" and str(d["head"].item()) == "omat"
    assert str(d["checkpoint_name"].item()) == "UMA-S-1.2.1"
    assert str(d["dtype"].item()) == "float32" and not bool(d["tf32"].item())
    assert np.array_equal(np.bincount(d["raw_sample_indices"], minlength=len(ref["measure"])), ref["measure"])
    schema = load_schema(schema_path)
    assert schema["identity"] == prior["condition_schema_id"]
    selected = [r for r in json.loads(selection.read_text())["selected_models"] if r["fit_seed"] == 20261106]
    assert len(selected) == 1
    checkpoint = Path(selected[0]["checkpoint"])
    assert sha256(checkpoint) == selected[0]["checkpoint_sha256"]
    # This is a mathematical support check, not a new small-cell energy study.
    with np.load(finite_parent, allow_pickle=False) as z: permutations = z["permutations"]
    with np.load(finite_data, allow_pickle=False) as z: occupancy12 = z["occupancy"]
    finite = validate_exhaustive_support(occupancy12, permutations)
    assert finite["representatives"] == 87 and finite["n_sites"] == 12
    write_json(output/"finite12_support.json", {**finite, "scope":"Existing masks only; no small-cell energy or size-transfer claim"})
    fp = json.loads(final_protocol.read_text());history_source = Path(fp["history_path"])
    assert sha256(history_source) == fp["history_sha256"]
    history = set(json.loads(history_source.read_text()))
    with np.load(final_pool, allow_pickle=False) as z:
        masks = z["masks"]
        assert str(z["parent_sha256"].item()) == sha256(parent)
    assert masks.shape[1] == 48 and np.isin(masks, [0,1]).all()
    history.update("".join(map(str, row)) for row in masks)
    # Freeze every currently registered parent-mask identity, including editor
    # seed/child queries, without reading their energies or using their outcomes.
    registry_path=root/"runs/conditioning_20261107/oracle/registry.json"
    registry_bytes=registry_path.read_bytes();registry=json.loads(registry_bytes)
    assert registry["parent_sha256"]==sha256(parent)
    registered=set()
    for entry in registry["physical_keys"].values():
        sid=entry["structure_id"]
        if not sid.startswith("mask48_") or len(sid)!=19:raise ValueError("Unexpected registry geometry identity")
        code=int(sid.split("_",1)[1],16)
        if not 0<=code<2**48:raise ValueError("Registry mask exceeds parent size")
        registered.add(format(code,"048b"))
    with np.load(editor_pool,allow_pickle=False) as z:
        assert str(z["parent_sha256"].item())==sha256(parent)
        editor_keys={"".join(map(str,row)) for row in z["masks"]}
    if not editor_keys<=registered:raise ValueError("Editor candidates/seeds must be registered before this follow-up history is frozen")
    history.update(registered)
    write_json(output/"registered_history.json",sorted(registered))
    write_json(output/"history.json", sorted(history))
    schedules, fractions, diagnostics = [], [], []
    labels = classify(schema, ref["n_target"], ref["energy_eV"]/ref["n_atoms"])
    for n in COUNTS:
        keep = ref["n_target"] == n
        assert int(ref["measure"][keep].sum()) == 8192
        names = supported_bins(schema, n)
        assert names == list(BINS)
        for temperature in TEMPERATURES:
            weight = static_weights(ref["energy_eV"][keep], ref["measure"][keep], temperature, measure_kind="uniform_raw_sample")
            pure = bin_probabilities(weight["probability"], labels[keep], names)
            diagnostics.append(dict(n_Li=n, temperature_K=temperature,
                particle_ESS=weight["microstate_or_particle_ess"], representative_ESS=weight["representative_ess"],
                largest_representative_mass=weight["maximum_representative_mass"],
                reference_mean_total_energy_eV=weight["mean_energy_eV"], reference_minimum_total_energy_eV=weight["E_ref_eV"],
                raw_reference_draws=int(weight["total_measure"]), reference_orbits=int(weight["representative_count"])))
            for policy, probabilities in [("static", pure), ("static_exploration", exploration_policy(pure, EXPLORATION))]:
                allocation = allocate_integer(probabilities, BUDGET)
                sid = f"temperature-{n}-{temperature:g}-{policy}"
                schedules.append(dict(schedule_id=sid, n_Li=n, temperature_K=temperature,
                    policy=policy, budget=BUDGET, counts=allocation, probabilities=probabilities))
                fractions.extend(dict(schedule_id=sid, n_Li=n, temperature_K=temperature, policy=policy,
                    requested_bin=b, requested_probability=probabilities[b], attempts=allocation[b], budget=BUDGET) for b in names)
    assert len(schedules)*BUDGET == 768
    paths = dict(required_gate=gate, parent=parent, reference=reference, reference_receipt=old_receipt,
        schema=schema_path, selection=selection, checkpoint=checkpoint, final_protocol=final_protocol,
        final_pool=final_pool, history_source=history_source, finite12_parent=finite_parent, finite12_data=finite_data,
        history=output/"history.json",editor_pool=editor_pool,registered_history=output/"registered_history.json")
    sources = {name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in paths.items()}
    protocol = dict(status="FROZEN_BEFORE_GENERATION_AND_SCORING", created_UTC=datetime.now(timezone.utc).isoformat(),
        namespace=NAMESPACE, hypothesis="Improved energy control may realize the existing coarse bin requests more faithfully; no temperature effect is presumed",
        exact_change="Use development-selected graph-context checkpoint seed20261106; retain the prior reference, schema and acquisition measure",
        controls="Pure static and epsilon=.1 uniform-over-supported-bins floor; common latent/uniform streams across T and matching bin prefixes across policies",
        decision_rule="Report exact requested and realized bins/energies; identical allocations/proposals across T are an honest null response, not evidence of physical equilibrium",
        sources=sources, fit_seed=20261106, counts=list(COUNTS), temperatures_K=list(TEMPERATURES),
        schedules=schedules, primary_attempts=768, fit_replicates=1, budget_per_schedule=BUDGET,
        exploration_fraction=EXPLORATION, exploration_definition="(1-epsilon)*reference bin mass+epsilon/3; integer allocation may differ from epsilon in realized non-low fraction",
        reference_measure="Uniform raw-mask sample collapsed for scoring with retained frequencies; no additional symmetry multiplicity",
        weighting="m_i*exp[-(E_i-Eref)/(kBT)], E in eV per whole cell, m_i raw draw frequency",
        physical_temperature_enters_decoder=False, no_new_reference=True, no_MACE=True, no_DFT=True,
        no_training=True, geometry_protocol=str(d["geometry_protocol"].item()), known_history_orbits=len(history),
        history_scope="Required-core pre-final history, every core final proposal, and all registered parent-mask identities including editor seeds/children; no editor energies read",
        registry_history_snapshot=dict(path=str(registry_path.resolve()),sha256_at_freeze=hashlib.sha256(registry_bytes).hexdigest(),
            registered_parent_orbits=len(registered),editor_pool_orbits=len(editor_keys),
            witness="Every editor pool geometry registered at freeze; immutable derived registered_history.json retained; live registry may append later"),
        expected_max_distinct_raw_draws=102, source_sha256=sha256(__file__),
        limits="One fitted model,32attempts/schedule; no fit-to-fit confidence claim; low reference ESS cannot be repaired by more proposals")
    write_json(output/"protocol.json", protocol)
    write_csv(output/"reference_diagnostics.csv", diagnostics)
    write_csv(output/"requested_allocations.csv", fractions)
    print(json.dumps(dict(status=protocol["status"],primary_attempts=768,history_orbits=len(history),
        allocations=[dict(n=s["n_Li"],T=s["temperature_K"],policy=s["policy"],counts=s["counts"]) for s in schedules]),indent=2))


def generate(output, device):
    require_compute_allocation()
    import torch
    from cathode_cvae.learning.generate_mask import load_model, resolve_request, attempt_records
    from cathode_cvae.learning.train_mask import canonicalize
    protocol = json.loads((output/"protocol.json").read_text());verify_sources(protocol)
    assert protocol["status"] == "FROZEN_BEFORE_GENERATION_AND_SCORING"
    gen = output/"generated";gen.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model,cfg = load_model(Path(protocol["sources"]["checkpoint"]["path"]), device)
    assert cfg["seed"] == 20261106 and cfg["latent_dim"] == 4 and cfg["graph_context"] and cfg["encoder_kind"] == "graph"
    with np.load(protocol["sources"]["parent"]["path"],allow_pickle=False) as z:
        inv = torch.tensor(np.argsort(z["permutations"],axis=1), device=device)
    schema = load_schema(protocol["sources"]["schema"]["path"])
    samples, fingerprints, maximums = {}, {}, {}
    started = time.time()
    for n in COUNTS:
        generator = torch.Generator(device=device).manual_seed(seed_for(NAMESPACE,n,"paired-latent-uniforms"))
        z = torch.randn((BUDGET,4),device=device,generator=generator)
        uniforms = torch.rand((BUDGET,48),device=device,generator=generator)
        for b in BINS:
            maximum = max(s["counts"][b] for s in protocol["schedules"] if s["n_Li"] == n)
            maximums[n,b] = maximum
            vector,*_ = resolve_request(cfg,n,b,None,schema)
            c = torch.tensor(vector,device=device).repeat(maximum,1)
            with torch.no_grad(): raw=model.sample(c,n,z=z[:maximum],uniforms=uniforms[:maximum])
            assert tuple(raw.shape) == (maximum,48) and bool(((raw==0)|(raw==1)).all()) and bool((raw.sum(1)==n).all())
            with torch.no_grad(): _,codes=canonicalize(raw,inv)
            keys=[format(int(code),"048b") for code in codes]
            samples[n,b]=(raw.cpu().numpy().astype(np.uint8),keys)
            fingerprints[n,b]=dict(latents_sha256=hashlib.sha256(z[:maximum].cpu().numpy().tobytes()).hexdigest(),
                uniforms_sha256=hashlib.sha256(uniforms[:maximum].cpu().numpy().tobytes()).hexdigest())
    rows,raws,candidate = [],[],{}
    for schedule in protocol["schedules"]:
        n=schedule["n_Li"]
        for b in BINS:
            count=schedule["counts"][b]
            if not count:continue
            raw,keys=samples[n,b];keys=keys[:count]
            ids=[f"mask48_{int(k,2):012x}" for k in keys]
            records=attempt_records(count,ids,keys,method="graph_context",fit_seed=20261106,
                sampling_seed=seed_for(NAMESPACE,n,"paired-latent-uniforms"),n_target=n,requested_bin=b,
                effective_model_class=b,schema_id=schema["identity"],
                checkpoint_sha256=protocol["sources"]["checkpoint"]["sha256"],
                parent_sha256=protocol["sources"]["parent"]["sha256"],geometry_protocol=protocol["geometry_protocol"],
                temperature_K=schedule["temperature_K"],allocation_policy=schedule["policy"],
                allocation_weight=schedule["probabilities"][b])
            for i,row in enumerate(records):
                row.update(attempt_id=f"{schedule['schedule_id']}-{b}-{i}",schedule_id=schedule["schedule_id"],
                    replicate_id=20261106,diagnostic="primary",raw_row=len(raws),paired_draw_index=i,
                    structure_id=ids[i],n_target=n,
                    **fingerprints[n,b]);rows.append(row);raws.append(raw[i]);candidate[keys[i]]=ids[i]
    assert len(rows) == protocol["primary_attempts"]
    keys=sorted(candidate);masks=np.array([[int(v) for v in k] for k in keys],dtype=np.uint8)
    np.savez_compressed(gen/"candidates.npz",masks=masks,occupancy=1-masks,
        structure_ids=np.array([candidate[k] for k in keys]),n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),
        parent_sha256=np.asarray(protocol["sources"]["parent"]["sha256"]))
    np.savez_compressed(gen/"raw_attempts.npz",occupancy=np.array(raws))
    write_csv(gen/"attempts.csv",rows)
    verify_sources(protocol)
    receipt=dict(status="FROZEN_BEFORE_SCORING",attempts=len(rows),unique_geometries=len(keys),
        raw_draws_before_reuse=sum(maximums.values()),actual_fit_replicates=1,common_RNG_across_temperature=True,
        unchanged_allocations_imply_bit_identical_proposals=True, physical_temperature_enters_decoder=False,
        protocol_sha256=sha256(output/"protocol.json"),source_sha256=sha256(__file__),
        elapsed_seconds=time.time()-started,device=device,torch=torch.__version__,
        outputs={name:sha256(gen/name) for name in ["attempts.csv","raw_attempts.npz","candidates.npz"]},
        no_proposal_energy_labels_read=True,no_energy_filtering=True)
    write_json(gen/"receipt.json",receipt);print(json.dumps(receipt,indent=2))


def analyze(output, score_path):
    require_compute_allocation()
    protocol=json.loads((output/"protocol.json").read_text());verify_sources(protocol)
    gen=output/"generated";generation=json.loads((gen/"receipt.json").read_text())
    for name,digest in generation["outputs"].items():assert sha256(gen/name)==digest
    with (gen/"attempts.csv").open() as f: attempts=list(csv.DictReader(f))
    with np.load(score_path,allow_pickle=False) as z:
        scores={k:z[k] for k in ["structure_ids","energy_eV","n_atoms","n_Li"]}
    ids=scores["structure_ids"].astype(str);assert len(ids)==len(set(ids))
    lookup={sid:i for i,sid in enumerate(ids)}
    missing=sorted({r["structure_id"] for r in attempts}-set(lookup))
    if missing:raise ValueError(f"Missing {len(missing)} prespecified exact-geometry scores; do not use cache-only outcome")
    schema=load_schema(protocol["sources"]["schema"]["path"])
    history=set(json.loads((output/"history.json").read_text()))
    for row in attempts:
        i=lookup[row["structure_id"]];n=int(row["n_target"])
        assert scores["n_atoms"][i]==144+n and scores["n_Li"][i]==n and np.isfinite(scores["energy_eV"][i])
        row.update(energy_eV=float(scores["energy_eV"][i]),energy_eV_atom=float(scores["energy_eV"][i]/(144+n)),
            measured_bin=str(classify(schema,np.array([n]),np.array([scores["energy_eV"][i]/(144+n)]))[0]),
            novel_to_frozen_history=row["canonical_mask"] not in history)
    results,confusion=[],[]
    for schedule in protocol["schedules"]:
        rows=[r for r in attempts if r["schedule_id"]==schedule["schedule_id"]]
        assert len(rows)==BUDGET
        useful={r["canonical_mask"] for r in rows if r["novel_to_frozen_history"] and r["requested_bin"]==r["measured_bin"]}
        counts=Counter(r["measured_bin"] for r in rows)
        results.append(dict(schedule_id=schedule["schedule_id"],n_Li=schedule["n_Li"],temperature_K=schedule["temperature_K"],
            policy=schedule["policy"],attempts=len(rows),unique_orbits=len({r["canonical_mask"] for r in rows}),
            useful_novel_inbin_orbits=len(useful),useful_yield=len(useful)/BUDGET,
            bin_hit_rate=sum(r["requested_bin"]==r["measured_bin"] for r in rows)/BUDGET,
            mean_total_energy_eV=float(np.mean([r["energy_eV"] for r in rows])),
            mean_energy_eV_atom=float(np.mean([r["energy_eV_atom"] for r in rows])),
            **{f"measured_{b}_fraction":counts[b]/BUDGET for b in BINS}))
        confusion.extend(dict(schedule_id=schedule["schedule_id"],n_Li=schedule["n_Li"],temperature_K=schedule["temperature_K"],
            policy=schedule["policy"],requested_bin=b,measured_bin=measured,
            attempts=sum(r["requested_bin"]==b and r["measured_bin"]==measured for r in rows)) for b in BINS for measured in BINS)
    paired=[]
    for n in COUNTS:
        for policy in ["static","static_exploration"]:
            schedules=[s for s in protocol["schedules"] if s["n_Li"]==n and s["policy"]==policy]
            sets=[[r["canonical_mask"] for r in attempts if r["schedule_id"]==s["schedule_id"]] for s in schedules]
            same_allocation=all(s["counts"]==schedules[0]["counts"] for s in schedules)
            same_proposals=all(keys==sets[0] for keys in sets)
            if same_allocation:assert same_proposals
            paired.append(dict(n_Li=n,policy=policy,identical_bin_allocations=same_allocation,
                bit_identical_paired_proposals=same_proposals))
    analysis=output/"analysis";analysis.mkdir(exist_ok=False)
    write_csv(analysis/"attempt_outcomes.csv",attempts);write_csv(analysis/"schedule_summary.csv",results)
    write_csv(analysis/"requested_vs_measured.csv",confusion);write_csv(analysis/"temperature_identity.csv",paired)
    receipt=dict(status="PASS",attempts=len(attempts),schedules=len(results),unique_scored_candidates=generation["unique_geometries"],
        missing_scores=0,fit_replicates=1,paired_temperature_checks=paired,
        protocol_sha256=sha256(output/"protocol.json"),generation_receipt_sha256=sha256(gen/"receipt.json"),
        score_path=str(score_path.resolve()),score_sha256=sha256(score_path),source_sha256=sha256(__file__),
        physical_cost_scope="Actual fresh labels/caches are in the integrating registry/cost receipt, not inferred from global collected provenance",
        interpretation="Static coarse-bin acquisition only; repeated temperatures sharing proposals are paired controls, not independent repetitions; no entropy, equilibrium or reactor validation")
    write_json(analysis/"receipt.json",receipt)
    plot(output,analysis,attempts,results,protocol)
    print(json.dumps(receipt,indent=2))


def plot(output,analysis,attempts,results,protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"low":"#268055","middle":"#e5a32a","high":"#b84c59"}
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharey=True,constrained_layout=True)
    for ci,n in enumerate(COUNTS):
        for pi,policy in enumerate(["static","static_exploration"]):
            ax=axes[ci,pi]
            for ti,t in enumerate(TEMPERATURES):
                schedule=next(s for s in protocol["schedules"] if s["n_Li"]==n and s["temperature_K"]==t and s["policy"]==policy)
                row=next(r for r in results if r["schedule_id"]==schedule["schedule_id"])
                bottom_request=bottom_realized=0
                for b in BINS:
                    requested=schedule["counts"][b]/BUDGET;realized=row[f"measured_{b}_fraction"]
                    ax.bar(ti-.18,requested,.32,bottom=bottom_request,color=colors[b],edgecolor="white",hatch="//",label=b if ci==pi==ti==0 else None)
                    ax.bar(ti+.18,realized,.32,bottom=bottom_realized,color=colors[b],edgecolor="white")
                    bottom_request+=requested;bottom_realized+=realized
            ax.set(xticks=range(4),xticklabels=[f"{t:.2f}" for t in TEMPERATURES],ylim=(0,1.05),
                title=f"nLi={n}, {policy}",xlabel="Outer allocation temperature (K)")
            if pi==0:ax.set_ylabel("Attempt fraction")
    axes[0,0].legend(loc="lower left",ncol=3,fontsize=8)
    fig.suptitle("Left/hatched: requested bins; right/solid: measured UMA bins\nSame random streams across T; identical requests give identical samples",fontsize=11)
    fig.savefig(analysis/"requested_vs_realized.pdf");fig.savefig(analysis/"requested_vs_realized.png",dpi=150);plt.close(fig)
    ref=load_reference(Path(protocol["sources"]["reference"]["path"]),Path(protocol["sources"]["parent"]["path"]),"uniform_raw_sample")
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,n in zip(axes,COUNTS):
        keep=ref["n_target"]==n;e=ref["energy_eV"][keep];order=np.argsort(e);minimum=e.min()
        for ti,t in enumerate(TEMPERATURES):
            w=static_weights(e,ref["measure"][keep],t,measure_kind="uniform_raw_sample")
            ax.plot(e[order]-minimum,np.cumsum(w["probability"][order]),lw=1,alpha=.7,label=f"Reference {t:g} K")
        for policy,color in [("static","#222222"),("static_exploration","#8635a0")]:
            rows=[r for r in attempts if int(r["n_target"])==n and float(r["temperature_K"])==TEMPERATURES[0] and r["allocation_policy"]==policy]
            es=np.sort([r["energy_eV"] for r in rows])-minimum
            ax.step(es,np.arange(1,len(es)+1)/len(es),where="post",lw=1.8,color=color,label=f"Generated {policy}")
        ax.set(title=f"nLi={n}",xlabel="Total-cell E − reference minimum (eV)",ylabel="Cumulative weight / attempt fraction",ylim=(0,1.02))
    axes[1].legend(fontsize=7,loc="lower right")
    fig.suptitle("Finite-reference static weights versus generated energies\nGenerated lines shown once: all four T reuse identical integer requests",fontsize=11)
    fig.savefig(analysis/"energy_distributions.pdf");fig.savefig(analysis/"energy_distributions.png",dpi=150);plt.close(fig)


def selftest():
    # Independent direct exponent arithmetic, including the repeated-particle ESS.
    e=np.array([0.,.12,.35]);m=np.array([1,2,1]);t=433.15
    w=m*np.exp(-(e-e.min())/(KB_EV_K*t));p=w/w.sum()
    got=static_weights(e,m,t,measure_kind="uniform_raw_sample")
    assert np.allclose(got["probability"],p,rtol=1e-14,atol=0)
    expanded=np.repeat(e,m);raw=np.exp(-expanded/(KB_EV_K*t));raw/=raw.sum()
    assert np.isclose(got["microstate_or_particle_ess"],1/np.dot(raw,raw),rtol=1e-14)
    assert allocate_integer({"low":1.,"middle":0.,"high":0.},32)=={"low":32,"middle":0,"high":0}
    assert allocate_integer(exploration_policy({"low":1.,"middle":0.,"high":0.},.1),32)=={"low":30,"middle":1,"high":1}
    assert not np.allclose(static_weights(e/168,m,t,measure_kind="uniform_raw_sample")["probability"],p)
    assert seed_for(NAMESPACE,12,"paired-latent-uniforms")==seed_for(NAMESPACE,12,"paired-latent-uniforms")
    print(json.dumps(dict(status="PASS",checks=6,scope="Independent exponent/frequency/ESS arithmetic, integer allocations, units guard and paired seed policy")))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","generate","analyze","selftest"])
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument("--output-dir",type=Path)
    p.add_argument("--device",default="cuda",choices=["cuda","cpu"]);p.add_argument("--scores",type=Path)
    p.add_argument("--editor-pool",type=Path,help="Already registered immutable editor seed/child candidate NPZ")
    a=p.parse_args();root=a.root.resolve();output=a.output_dir or root/"reports/conditioning_20261107/temperature_followup"
    if a.action=="prepare":
        if a.editor_pool is None:p.error("--editor-pool must be registered before temperature history/protocol preparation")
        prepare(root,output,a.editor_pool.resolve())
    elif a.action=="generate":generate(output,a.device)
    elif a.action=="analyze":
        if a.scores is None:p.error("--scores exact-geometry collected UMA NPZ is required")
        analyze(output,a.scores)
    else:selftest()


if __name__=="__main__":main()
