"""One GPU, one bounded archived Li event: fixed-cell UMA endpoints and CI-NEB.

Root launches this only after optional editor work. No MD, DFT input generation,
or submission occurs here. The process has its own wall/call bound; the launching
owner must also cap allocated GPU-device-hours, including startup and idle time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import signal
import time
import traceback

import ase
import numpy as np
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import FIRE

from neb_common import (BudgetExpired, CallLedger, CountingCalculator, attach_calculators, atoms_from,
    digest, discrete_barriers, export_images, load_npz, mapping, read_json, snapshot, validate_endpoint_event, write_json)


def fmax(forces):
    forces = np.asarray(forces).reshape(-1, 3)
    if not np.isfinite(forces).all(): raise ValueError("Nonfinite projected or physical force; preserve failure")
    return float(np.linalg.norm(forces, axis=1).max())


def safe_idpp(neb, **kwargs):
    """Do not retain IDPP fake-label calculators after a budget interruption."""
    original = [im.calc for im in neb.images]
    try:
        return idpp_interpolate(neb, **kwargs)
    finally:
        for im, calc in zip(neb.images, original): im.calc = calc


def resample_band(images, interior):
    positions = np.asarray([im.positions for im in images])
    lengths = np.linalg.norm(np.diff(positions, axis=0).reshape(len(images)-1, -1), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    if arc[-1] <= 0: raise ValueError("Cannot refine a collapsed zero-length band")
    unique = np.r_[True, np.diff(arc) > 1e-14]
    targets = np.linspace(0, arc[-1], interior+2)
    flat = positions.reshape(len(images), -1)
    interpolated = np.asarray([np.interp(targets, arc[unique], column[unique]) for column in flat.T]).T
    result = [images[0].copy() for _ in targets]
    for im, values in zip(result, interpolated): im.positions[:] = values.reshape(-1, 3)
    return result


def verify_prepared(prepared):
    protocol = read_json(prepared / "protocol.json")
    if protocol.get("status") != "PREPARED_NO_MLIP_EVALUATION" or protocol["maximum_events"] != 2:
        raise ValueError("Not the frozen two-event NEB protocol")
    if ase.__version__ != protocol["installed_ASE"]: raise ValueError("Installed ASE changed since preparation")
    for obj in (NEB, FIRE, idpp_interpolate):
        if digest(inspect.getsourcefile(obj)) != protocol["installed_API"][obj.__name__]["source_sha256"]:
            raise ValueError("Installed ASE implementation differs from the prepared protocol")
    for name, sha in protocol["script_hashes"].items():
        if digest(Path(__file__).parent / name) != sha: raise ValueError(f"Prepared NEB source changed: {name}")
    gate = protocol["required_report_gate"]
    if digest(gate["path"]) != gate["sha256"]: raise ValueError("Required-report gate changed")
    for path, sha in protocol["source_hashes"].items():
        if path.endswith(("calculators.py", "registry.py")) and digest(path) != sha:
            raise ValueError("Pinned calculator adapter/registry changed")
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=2400)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    prepared, out = args.prepared.resolve(), args.output_dir.resolve()
    protocol = verify_prepared(prepared)
    if args.budget_seconds <= 0 or args.budget_seconds > protocol["per_event_wall_seconds"]:
        raise ValueError("Per-event wall budget may not exceed the frozen 2400 seconds")
    event = next(e for e in protocol["events"] if e["event_id"] == args.event)
    if args.refine and args.event != protocol["refinement"]["event_id"]:
        raise ValueError("The image/tolerance refinement is bounded to the first event")
    if out.exists(): raise FileExistsError("Preserve existing NEB attempts, including failures")
    out.mkdir(parents=True, exist_ok=False)
    ledger = CallLedger(out / "calculator_calls.jsonl", args.budget_seconds,
        max_calls=protocol["maximum_calculator_EF_attempts_per_event"], start=started)
    def stop(signum, frame): raise BudgetExpired(f"External termination signal {signum}; save bounded failure")
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    result = dict(status="STARTED", started_UTC=started_utc, event_id=args.event, event=event,
        protocol_sha256=digest(prepared / "protocol.json"), script_sha256=digest(__file__),
        job_id=os.environ.get("SLURM_JOB_ID"), host=__import__("socket").gethostname(),
        GPU_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), device=args.device,
        budget_seconds=args.budget_seconds, actual_backbone_forward_count="not instrumented",
        scope="Fixed-cell UMA potential-energy feasibility pilot; not finite-temperature free energy or DFT validation",
        forward_barrier_eV=None, reverse_barrier_eV=None, image_tolerance_check="not attempted")
    parent_path = Path(protocol["parent"]["path"])
    if digest(parent_path) != protocol["parent"]["sha256"]: raise ValueError("Exact parent changed")
    parent = load_npz(parent_path)
    inputs = {}
    for name in ("initial", "final"):
        binding = event["endpoint_inputs"][name]; path = prepared / binding["path"]
        if digest(path) != binding["sha256"]: raise ValueError("Prepared endpoint changed")
        inputs[name] = load_npz(path)
    parent_indices = inputs["initial"]["parent_indices"]
    if not np.array_equal(parent_indices, inputs["final"]["parent_indices"]): raise ValueError("Endpoint identities differ")
    endpoints = [atoms_from(inputs[name]) for name in ("initial", "final")]
    if not np.array_equal(endpoints[0].cell, endpoints[1].cell) or not np.array_equal(endpoints[0].numbers, endpoints[1].numbers):
        raise ValueError("Standard NEB requires common fixed cell and atom order")
    current_images = endpoints; current_neb = None
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        ledger.check()
        inner, teacher = make_calculator("uma", args.checkpoint, device=args.device, threads=args.threads)
        result["teacher"] = teacher; ledger.check()
        endpoint_results = []
        for index, atom in enumerate(endpoints):
            ledger.phase = f"endpoint_{index}"
            atom.calc = CountingCalculator(inner, ledger, f"endpoint:{index}")
            settings = protocol["endpoint_relaxation"]
            optimizer = FIRE(atom, logfile=str(out/f"endpoint_{index}.log"), trajectory=str(out/f"endpoint_{index}.traj"),
                dt=settings["dt_ASE"], dtmax=settings["dtmax_ASE"], maxstep=settings["maxstep_A"])
            optimizer.attach(ledger.check, interval=1)
            converged = bool(optimizer.run(fmax=settings["Fmax_eV_A"], steps=settings["max_steps_per_endpoint"]))
            residual = fmax(atom.get_forces()); endpoint_map = mapping(atom, parent, parent_indices)
            endpoint_results.append(dict(converged=converged, optimizer_steps=optimizer.nsteps,
                physical_Fmax_eV_A=residual, energy_eV=float(atom.get_potential_energy()), mapping=endpoint_map))
            snapshot([atom], out/f"endpoint_{index}.npz", parent_indices=parent_indices)
            write_json(out / "endpoint_results.json", endpoint_results)
            if not converged or residual > settings["Fmax_eV_A"]:
                result["status"] = "ENDPOINT_FORCE_TARGET_NOT_MET"; break
        result["endpoints"] = endpoint_results
        if len(endpoint_results) != 2 or any(not e["converged"] for e in endpoint_results):
            return 2
        valid, reason = validate_endpoint_event(endpoint_results[0]["mapping"], endpoint_results[1]["mapping"],
            event["expected_initial_assignment"], event["expected_final_assignment"], event["migrating_atom_index"],
            event["source_parent_Li_site"], event["destination_parent_Li_site"])
        result["endpoint_event_validation"] = dict(valid=valid, reason=reason)
        if not valid:
            result["status"] = "FAILED_EVENT_ENDPOINT_QUENCH"; return 2
        export_images(endpoints, out/"quenched_endpoints", parent_indices)
        settings = protocol["initial_band"]
        images = [endpoints[0]] + [endpoints[0].copy() for _ in range(settings["interior_images"])] + [endpoints[1]]
        # The source MD coordinates are already unwrapped. Direct interpolation
        # preserves its observed hop image, including half-cell ambiguities.
        neb = NEB(images, k=settings["k_eV_A2"], method=settings["method"], climb=False,
            parallel=False, remove_rotation_and_translation=False, allow_shared_calculator=False)
        neb.interpolate(method="linear", mic=False, apply_constraint=False)
        current_images, current_neb = images, neb
        # ASE3.26 restores real calculators only on normal IDPP return. Restore
        # them ourselves on interruption, so fake IDPP labels cannot be saved
        # under UMA provenance in a bounded-failure receipt.
        idpp_steps = {}; idpp_started = time.monotonic()
        def bounded_idpp_optimizer(*positional, **keywords):
            opt = FIRE(*positional, **keywords, dt=settings["dt_ASE"], dtmax=settings["dtmax_ASE"], maxstep=settings["maxstep_A"])
            opt.attach(ledger.check, interval=1)
            opt.attach(lambda: idpp_steps.update(optimizer_steps=opt.nsteps), interval=1)
            return opt
        safe_idpp(neb, mic=True, steps=settings["IDPP_max_steps"], fmax=settings["IDPP_optimizer_fmax"],
            optimizer=bounded_idpp_optimizer, traj=str(out/"initial_idpp.traj"), log=str(out/"initial_idpp.log"))
        result["IDPP_initialization"] = dict(status="COMPLETE", elapsed_seconds=time.monotonic()-idpp_started,
            **idpp_steps, MLIP_calculator_calls=0, optimizer_force_threshold=settings["IDPP_optimizer_fmax"],
            threshold_scope=settings["IDPP_threshold_scope"])
        attach_calculators(images, inner, ledger, "band5")
        current_images, current_neb = images, neb

        def run_band(label, neb_object, band, target, steps):
            ledger.phase = label
            opt = FIRE(neb_object, logfile=str(out/(label+".log")), trajectory=str(out/(label+".traj")),
                dt=settings["dt_ASE"], dtmax=settings["dtmax_ASE"], maxstep=settings["maxstep_A"])
            def observe():
                ledger.check()
                projected = neb_object.get_forces()
                if opt.nsteps % 10 == 0:
                    snapshot(band, out/(label+"_latest.npz"), projected_forces=projected, parent_indices=parent_indices)
                with (out/"optimizer_progress.jsonl").open("a") as f:
                    import json
                    f.write(json.dumps(dict(phase=label, optimizer_step=opt.nsteps, NEB_projected_Fmax_eV_A=fmax(projected),
                        physical_image_Fmax_eV_A=[fmax(im.calc.results["forces"]) for im in band],
                        total_calculator_EF_attempts=ledger.attempts)) + "\n")
            opt.attach(observe, interval=1)
            converged = bool(opt.run(fmax=target, steps=steps))
            projected = neb_object.get_forces()
            saved = snapshot(band, out/(label+"_final.npz"), projected_forces=projected, parent_indices=parent_indices)
            climbing_index = int(neb_object.imax) if neb_object.climb else None
            return dict(converged=converged, optimizer_steps=opt.nsteps, target_Fmax_eV_A=target,
                NEB_projected_Fmax_eV_A=fmax(projected), physical_image_Fmax_eV_A=[fmax(f) for f in saved["forces_eV_A"]],
                climbing_image_index=climbing_index,
                physical_climbing_image_Fmax_eV_A=fmax(saved["forces_eV_A"][climbing_index]) if climbing_index is not None else None,
                energy_eV=saved["energy_eV"].tolist(), **discrete_barriers(saved["energy_eV"]))

        result["initial_band"] = run_band("initial_band", neb, images, settings["initial_Fmax_eV_A"], settings["initial_max_steps"])
        neb.climb = True
        result["climbing_band"] = run_band("climbing_band", neb, images, settings["climbing_Fmax_eV_A"], settings["climbing_max_steps"])
        climb = result["climbing_band"]
        if not climb["converged"]:
            result["status"] = "NEB_FORCE_TARGET_NOT_MET"; return 2
        if not climb["internal_peak"]:
            result["status"] = "NO_RESOLVED_INTERNAL_SADDLE"; return 2
        result.update(status="CONVERGED_NUMERICALLY_UNVALIDATED", forward_barrier_eV=climb["discrete_forward_span_eV"],
            reverse_barrier_eV=climb["discrete_reverse_span_eV"])
        export_images(images, out/"DFT_candidates_UMA_band5", parent_indices)
        if args.refine:
            refinement = protocol["refinement"]
            if ledger.deadline-time.monotonic() < refinement["minimum_remaining_wall_seconds"]:
                result["image_tolerance_check"] = "DEFERRED_REMAINING_WALL_BUDGET"
            else:
                refined = resample_band(images, refinement["interior_images"])
                attach_calculators(refined, inner, ledger, "band7")
                refined_neb = NEB(refined, k=settings["k_eV_A2"], method=settings["method"], climb=True,
                    parallel=False, remove_rotation_and_translation=False, allow_shared_calculator=False)
                current_images, current_neb = refined, refined_neb
                result["refinement"] = run_band("refined_band7", refined_neb, refined,
                    refinement["climbing_Fmax_eV_A"], refinement["max_steps"])
                check = result["refinement"]
                result["image_tolerance_check"] = "COMPLETE" if check["converged"] else "FORCE_TARGET_NOT_MET"
                result["forward_barrier_change_eV"] = check["discrete_forward_span_eV"]-climb["discrete_forward_span_eV"]
                result["reverse_barrier_change_eV"] = check["discrete_reverse_span_eV"]-climb["discrete_reverse_span_eV"]
                export_images(refined, out/"DFT_candidates_UMA_band7", parent_indices)
        return 0
    except BudgetExpired as error:
        result.update(status="BOUNDED_BUDGET_TERMINATION", error=str(error)); return 3
    except Exception as error:
        result.update(status="FAILED_PRESERVED", exception_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()); return 2
    finally:
        result["call_accounting"] = ledger.summary()
        result["completed_UTC"] = datetime.now(timezone.utc).isoformat()
        result["barrier_units"] = "eV per marked hop; potential energy, not activation free energy"
        # No new calculator access during failure serialization. Unknown labels
        # remain NaN in NPZ, with an explicit same-geometry cache-validity array.
        snapshot(current_images, out/"last_state.npz", parent_indices=parent_indices)
        result["DFT_policy"] = "POSCAR/CIF/NPZ candidates only. No VASP input decks, jobs, or DFT calls."
        write_json(out/"receipt.json", result)
        print(__import__("json").dumps(dict(status=result["status"], event=args.event,
            actual_calculator_EF_attempts=ledger.attempts, forward_barrier_eV=result["forward_barrier_eV"]), indent=2))


if __name__ == "__main__": raise SystemExit(main())
