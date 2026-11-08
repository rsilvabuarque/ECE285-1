"""Optional MACE single-point disagreement on a frozen UMA-optimized band.

This is deliberately NOT a MACE NEB or an independently optimized MACE barrier.
Only the already declared first event is eligible, after its UMA CI-NEB converges.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from neb_common import (BudgetExpired, CallLedger, CountingCalculator, atoms_from, digest,
    discrete_barriers, load_npz, read_json, snapshot, write_json)
from neb_run import verify_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--uma-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    start = time.monotonic(); protocol = verify_prepared(args.prepared)
    if not 0 < args.budget_seconds <= protocol["optional_MACE_profile_wall_seconds"]:
        raise ValueError("Optional MACE profile exceeds the frozen wall budget")
    uma = read_json(args.uma_run/"receipt.json")
    if uma["event_id"] != protocol["refinement"]["event_id"] or not uma.get("climbing_band", {}).get("converged"):
        raise ValueError("Only the first numerically converged UMA path is eligible")
    source = args.uma_run/"climbing_band_final.npz"
    band = load_npz(source)
    if not band["labels_match_geometry"].all(): raise ValueError("UMA path contains unknown same-geometry labels")
    if args.output_dir.exists(): raise FileExistsError("Preserve the existing MACE profile, including failures")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger = CallLedger(args.output_dir/"calculator_calls.jsonl", args.budget_seconds, max_calls=12, start=start)
    ledger.phase = "MACE_single_points_on_UMA_band"
    def stop(signum, frame): raise BudgetExpired(f"External termination signal {signum}")
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    images = [atoms_from(band, positions=p) for p in band["positions_A"]]
    result = dict(status="STARTED", started_UTC=datetime.now(timezone.utc).isoformat(), event_id=uma["event_id"],
        source_UMA_band=dict(path=str(source),sha256=digest(source)),
        source_UMA_receipt=dict(path=str(args.uma_run/"receipt.json"),sha256=digest(args.uma_run/"receipt.json")),
        scope="MACE single-point potential-energy profile on fixed UMA-optimized coordinates, NOT an independently optimized MACE barrier",
        independently_optimized_MACE_barrier_eV=None, optimization_updates=0, script_sha256=digest(__file__))
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        inner, teacher = make_calculator("mace", args.checkpoint, device=args.device, threads=args.threads)
        result["teacher"] = teacher
        energies, forces = [], []
        for i, image in enumerate(images):
            image.calc = CountingCalculator(inner, ledger, f"UMA_band_image:{i}")
            energies.append(float(image.get_potential_energy())); forces.append(image.get_forces())
        result.update(status="COMPLETE_SINGLE_POINT_DISAGREEMENT_ONLY", energy_eV=energies,
            physical_MACE_Fmax_per_image_eV_A=np.linalg.norm(forces, axis=2).max(1).tolist(),
            profile_spans_not_MACE_barriers=discrete_barriers(energies),
            UMA_energy_eV=band["energy_eV"].tolist(),
            disagreement_in_endpoint_referenced_profile_eV=((np.asarray(energies)-energies[0])-(band["energy_eV"]-band["energy_eV"][0])).tolist())
        return 0
    except BudgetExpired as error:
        result.update(status="BOUNDED_BUDGET_TERMINATION",error=str(error)); return 3
    except Exception as error:
        result.update(status="FAILED_PRESERVED",error=str(error),traceback=traceback.format_exc()); return 2
    finally:
        snapshot(images,args.output_dir/"MACE_on_UMA_band.npz",parent_indices=band["parent_indices"])
        result["call_accounting"] = ledger.summary(); result["completed_UTC"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir/"receipt.json",result)
        print(__import__("json").dumps(dict(status=result["status"],actual_calculator_EF_attempts=ledger.attempts),indent=2))
        

if __name__ == "__main__": raise SystemExit(main())
