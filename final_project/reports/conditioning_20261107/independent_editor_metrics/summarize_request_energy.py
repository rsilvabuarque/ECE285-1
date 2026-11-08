"""Posthoc request-specific interpretation of independently audited editor pairs.

No new labels, model calls, training, thresholds or selection. HIGH requests may
legitimately seek higher energies; pooled energy differences are not a dedicated
low-energy-search comparison. This reads only the immutable independent tables.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def write(path,values):
    if Path(path).exists():raise FileExistsError("Preserve prior posthoc analysis")
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",type=Path,required=True);a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    directory=a.directory;paths={name:directory/name for name in ["receipt.json","attempt_outcomes.csv","paired_energy_outcomes.csv"]}
    receipt=json.loads(paths["receipt.json"].read_text())
    if receipt["status"]!="PASS":raise ValueError("Independent editor check must precede posthoc interpretation")
    pairs=rows(paths["paired_energy_outcomes.csv"]);outcomes=rows(paths["attempt_outcomes.csv"])
    by_attempt={r["attempt_id"]:r for r in outcomes}
    for r in pairs:
        if r["both_energies_known"]!="True":raise ValueError("Unknown pair cannot be omitted from this completed audit")
    def summarize(selected):
        delta=np.asarray([float(r["learned_minus_mutation_meV_atom"]) for r in selected]);method_delta={}
        for method in ("learned_edit","mutation"):
            method_delta[method]=np.asarray([float(by_attempt[r["paired_trial_id"]+"-"+method]["delta_energy_meV_atom"]) for r in selected])
        if not np.allclose(delta,method_delta["learned_edit"]-method_delta["mutation"],atol=1e-10,rtol=1e-10):
            raise ValueError("Paired child difference does not cancel its actual common seed")
        return dict(all_paired_attempts=len(selected),known_pairs=len(delta),
            mean_learned_minus_mutation_meV_atom=float(delta.mean()),median_learned_minus_mutation_meV_atom=float(np.median(delta)),
            learned_lower_pairs=int(np.count_nonzero(delta<0)),equal_energy_pairs=int(np.count_nonzero(delta==0)),
            mutation_lower_pairs=int(np.count_nonzero(delta>0)),learned_lower_fraction_all_attempts=float(np.count_nonzero(delta<0)/len(selected)),
            mutation_lower_fraction_all_attempts=float(np.count_nonzero(delta>0)/len(selected)),
            mean_learned_child_minus_seed_meV_atom=float(method_delta["learned_edit"].mean()),
            mean_mutation_child_minus_seed_meV_atom=float(method_delta["mutation"].mean()))
    cells=[];fit=[];summary=[]
    for b in ("low","middle","high"):
        selected=[r for r in pairs if r["requested_bin"]==b]
        summary.append(dict(requested_bin=b,**summarize(selected)))
        for seed in (20261106,20261107,20261108):
            ss=[r for r in selected if int(r["fit_seed"])==seed]
            fit.append(dict(requested_bin=b,fit_seed=seed,**summarize(ss)))
            for n in (12,24,36):
                cc=[r for r in ss if int(r["n_Li"])==n]
                if len(cc)!=128:raise ValueError("Conditional cell lost its prespecified128 paired attempts")
                cells.append(dict(requested_bin=b,fit_seed=seed,n_Li=n,**summarize(cc)))
    write(directory/"requested_bin_energy_summary.csv",summary)
    write(directory/"requested_bin_energy_by_fit.csv",fit)
    write(directory/"requested_bin_energy_cells.csv",cells)
    write(directory/"LOW_request_paired_energy_cells.csv",[r for r in cells if r["requested_bin"]=="low"])
    output=dict(status="POSTHOC_INTERPRETATION_ONLY",completed_UTC=datetime.now(timezone.utc).isoformat(),
        source_hashes={name:sha(path) for name,path in paths.items()},script_sha256=sha(__file__),
        new_MLIP_calls=0,new_training_updates=0,selection_changed=False,all_attempts_retained=True,
        summary=summary,LOW_by_fit=[r for r in fit if r["requested_bin"]=="low"],
        interpretation="Requested HIGH and MIDDLE classes must be assessed under their intended targets. Overall energy lowering is not alone evidence about LOW search. LOW original-q20 useful yield and LOW paired energy differences are distinct observables.",
        uncertainty="Descriptive three-fit comparisons; no confidence interval based on treating correlated proposals as independent")
    target=directory/"requested_bin_energy_posthoc.json"
    with target.open("x") as f:json.dump(output,f,indent=2);f.write("\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":main()
