"""Summarize saved NEB outcomes, including failures; no calculator is loaded."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from neb_common import digest, load_npz, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    if args.output_dir.exists(): raise FileExistsError("Preserve the previous NEB summary")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipts = sorted(args.run_root.glob("*/receipt.json"))
    if not receipts: raise ValueError("No NEB/profile receipts to summarize")
    sources, rows, profiles = {}, [], []
    for path in receipts:
        receipt = read_json(path); sources[str(path)] = digest(path)
        counter = receipt.get("call_accounting", {})
        if "source_UMA_band" in receipt:
            rows.append(dict(event=receipt["event_id"], teacher="mace", method="single_points_on_UMA_path",
                status=receipt["status"], forward_barrier_eV="", reverse_barrier_eV="",
                calculator_EF_attempts=counter.get("actual_calculator_EF_attempts", ""),
                process_wall_seconds=counter.get("elapsed_seconds", "")))
            data_path = path.parent/"MACE_on_UMA_band.npz"
            if data_path.exists(): profiles.append((receipt["event_id"],"MACE SP on UMA band (not MACE NEB)",data_path,False))
            continue
        endpoints = receipt.get("endpoints", [])
        band, refined = receipt.get("climbing_band", {}), receipt.get("refinement", {})
        rows.append(dict(event=receipt["event_id"],teacher="uma",method="fixed_cell_CI_NEB",
            status=receipt["status"],forward_barrier_eV=receipt.get("forward_barrier_eV"),reverse_barrier_eV=receipt.get("reverse_barrier_eV"),
            endpoint_0_Fmax_eV_A=endpoints[0]["physical_Fmax_eV_A"] if endpoints else "",
            endpoint_1_Fmax_eV_A=endpoints[1]["physical_Fmax_eV_A"] if len(endpoints)>1 else "",
            projected_NEB_Fmax_eV_A=band.get("NEB_projected_Fmax_eV_A", ""),
            physical_climbing_image_Fmax_eV_A=band.get("physical_climbing_image_Fmax_eV_A", ""),
            climbing_optimizer_steps=band.get("optimizer_steps", ""),
            image_tolerance_check=receipt.get("image_tolerance_check", ""),
            refined_projected_NEB_Fmax_eV_A=refined.get("NEB_projected_Fmax_eV_A", ""),
            forward_barrier_change_eV=receipt.get("forward_barrier_change_eV", ""),
            reverse_barrier_change_eV=receipt.get("reverse_barrier_change_eV", ""),
            calculator_EF_attempts=counter.get("actual_calculator_EF_attempts", ""),
            process_wall_seconds=counter.get("elapsed_seconds", "")))
        for filename,label,key in (("climbing_band_final.npz","UMA 5 interior images","climbing_band"),
                                   ("refined_band7_final.npz","UMA 7 interior images; tighter target","refinement")):
            data_path = path.parent/filename
            if data_path.exists(): profiles.append((receipt["event_id"],label,data_path,bool(receipt.get(key,{}).get("converged"))))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir/"outcomes.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if profiles:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        events = sorted({event for event,_,_,_ in profiles})
        fig,axes = plt.subplots(1,len(events),figsize=(6*len(events),4),squeeze=False)
        for event,label,path,converged in profiles:
            data = load_npz(path); sources[str(path)] = digest(path)
            energy = data["energy_eV"]
            if not np.isfinite(energy).all(): continue
            positions = data["positions_A"]
            arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0).reshape(len(positions)-1,-1),axis=1))]
            ax = axes[0,events.index(event)]
            ax.plot(arc,energy-energy[0],"o-" if converged else "o--",label=label)
            ax.set(title=event,xlabel="Configuration-space arclength (Å)",ylabel="Energy relative to own initial image (eV/hop)")
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(args.output_dir/"profiles.pdf");fig.savefig(args.output_dir/"profiles.png",dpi=170);plt.close(fig)
    write_json(args.output_dir/"receipt.json",dict(status="SAVED_OUTCOMES_SUMMARIZED",source_hashes=sources,
        rows=len(rows),new_model_calls=0,script_sha256=digest(__file__),
        cost_scope="process_wall_seconds is not allocated GPU-device-hours; use root's Slurm cost receipt for allocations/startup/idle",
        physical_scope="MLIP potential-energy pilot. Failed/collapsed/unconverged paths remain visible. MACE curves are single-point disagreements only."))
    return 0


if __name__ == "__main__": raise SystemExit(main())
