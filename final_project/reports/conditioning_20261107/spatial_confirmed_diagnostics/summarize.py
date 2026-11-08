"""Descriptive architecture diagnostics with equal count and fit-seed weights."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cathode_cvae.runtime import require_compute_allocation


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows, fields, *, group_fields=("method","comparison","replicate_id","n_Li")):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in group_fields)].append(r)
    cells=[]
    for key, rr in sorted(groups.items()):
        cell=dict(zip(group_fields,key))
        for name,func in fields.items():cell[name]=float(np.mean([func(r) for r in rr]))
        cells.append(cell)
    seeds=defaultdict(list)
    for r in cells:seeds[(r["method"],r.get("comparison",""),r["replicate_id"])].append(r)
    seed_values=[]
    for (method,comparison,seed), rr in sorted(seeds.items()):
        record=dict(method=method,comparison=comparison,replicate_id=seed)
        for name in fields:record[name]=float(np.mean([r[name] for r in rr]))
        seed_values.append(record)
    methods=defaultdict(list)
    for r in seed_values:methods[(r["method"],r["comparison"])].append(r)
    result=[]
    for (method,comparison), rr in sorted(methods.items()):
        record=dict(method=method,comparison=comparison,design_seed_count=len(rr),seed_values=rr)
        for name in fields:record[name]=float(np.mean([r[name] for r in rr]))
        result.append(record)
    return result


def main():
    require_compute_allocation()
    root=Path(__file__).resolve().parent
    output=root/"family_diagnostic_summary.json"
    if output.exists():raise FileExistsError("Preserve existing diagnostic summary")
    tables={name:root/(name+".csv") for name in ["paired_conditions","paired_latents","posterior_prior_pairs","selected_latent_diagnostics"]}
    pair_fields=dict(mean_shift_meV_atom=lambda r:1000*float(r["delta_E_eV_atom"]),
        mean_absolute_shift_meV_atom=lambda r:1000*abs(float(r["delta_E_eV_atom"])),
        fraction_positive_shift=lambda r:float(float(r["delta_E_eV_atom"])>0),
        mean_Hamming=lambda r:float(r["raw_Hamming"]),same_orbit_fraction=lambda r:float(r["same_orbit"]=="True"))
    conditions=describe(read(tables["paired_conditions"]),pair_fields)
    latent=describe(read(tables["paired_latents"]),pair_fields)
    posterior=describe(read(tables["posterior_prior_pairs"]),dict(
        posterior_Hamming=lambda r:float(r["posterior_target_Hamming"]),
        prior_Hamming=lambda r:float(r["prior_target_Hamming"]),
        posterior_target_energy_MAE_meV_atom=lambda r:1000*abs(float(r["posterior_minus_target_eV_atom"])),
        prior_target_energy_MAE_meV_atom=lambda r:1000*abs(float(r["prior_minus_target_eV_atom"]))))
    fits=[r for r in read(tables["selected_latent_diagnostics"]) if float(r["selection_beta"])==.1 and r["methods"]]
    selected=[dict(method=r["methods"],fit=r["fit"],fit_seed=int(r["fit_seed"]),
        latent_dim=int(r["latent_dim"]),parameters=int(r["parameter_count"]),
        validation_reconstruction_nats=float(r["val_reconstruction"]),total_KL_nats=float(r["total_KL_nats"]),
        active_at_0p001=int(r["active_at_0.001"]),active_at_0p01=int(r["active_at_0.01"]),
        active_at_0p1=int(r["active_at_0.1"])) for r in fits]
    result=dict(status="PASS",scope="Confirmed DEVELOPMENT only",conditions=conditions,latents=latent,
        posterior_prior=posterior,selected_fit_latent_diagnostics=selected,
        weighting="Average proposal observables within each count, then equal counts within each design seed, then equal design seeds. Seeds are shown; no proposal-level confidence interval.",
        energy_units="meV/atom; positive condition shift means the second requested condition produced higher UMA energy",
        active_definition="Pooled validation posterior-mean variance greater than the descriptive dimensionless cutoff; neither physical accuracy nor a formal information test",
        input_hashes={p.name:sha(p) for p in tables.values()},source_sha256=sha(Path(__file__)))
    with output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
