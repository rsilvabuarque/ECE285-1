"""Persistent one-GPU fixed-cell relaxation worker; never creates allocations.

Terminal failures are evidence, not automatically retried. Run the frozen pilot
task list first, inspect measured throughput, then authorize the all-task list.
Each result binds literal initial geometry, teacher, endpoint protocol and roles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback

import numpy as np

from paired_common import (AttemptWallLimit, atomic_json, atomic_npz, canonical,
    check_prepared, counted_calculator, geometry_hash, identity, load_npz,
    mapping_diagnostics, occupancy_from_key, physical_key, sha256)


def run_attempt(base, parent, task, context, output, *, synchronize=lambda: None):
    """One bounded attempt. Failure serialization makes no calculator calls."""
    from ase.optimize import FIRE
    from cathode_cvae.data import deletion_atoms
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    occupancy = occupancy_from_key(task["canonical_vacancy_key"])
    key, _ = canonical(occupancy, parent["permutations"])
    if key != task["canonical_vacancy_key"]:
        raise ValueError("Queue input must use the canonical parent orientation")
    atoms, slots = deletion_atoms(parent, occupancy)
    initial_positions = atoms.positions.copy()
    initial_geometry = geometry_hash(atoms.numbers, atoms.positions, atoms.cell.array)
    numerical = context["optimizer"]
    counter = counted_calculator(base, deadline=time.monotonic() + numerical["wall_cap_seconds"],
                                 synchronize=synchronize)
    atoms.calc = counter
    trace, snapshots, trajectory = [], [], []
    initial = None
    opt = None
    status = "error"
    error = None
    interrupted = False

    def capture(force=False):
        if counter.last is None:
            return
        last = counter.last
        step = int(opt.nsteps) if opt is not None else 0
        same_geometry = bool(trajectory and np.array_equal(trajectory[-1], last["positions_A"]))
        fm = float(np.linalg.norm(last["forces_eV_A"], axis=1).max())
        if not same_geometry:
            trajectory.append(last["positions_A"].copy())
            row = [step, last["energy_eV"], fm, time.perf_counter() - started,
                   counter.attempts, counter.successes, counter.failures]
            trace.append(row)
            with (output / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        if (force or step % numerical["snapshot_stride"] == 0) and (
                not snapshots or not np.array_equal(snapshots[-1][1], last["positions_A"])):
            snapshots.append((step, last["positions_A"].copy(), last["forces_eV_A"].copy(),
                              last["stress_eV_A3"].copy(), last["energy_eV"]))

    class PhysicalFIRE(FIRE):
        def converged(self, gradient):
            fmax = float(np.linalg.norm(self.atoms.get_forces(), axis=1).max())
            return fmax < self.fmax

    try:
        atoms.get_potential_energy()
        initial = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in counter.last.items()}
        opt = PhysicalFIRE(atoms, logfile=str(output / "optimizer.log"),
                           maxstep=numerical["maxstep_A"])
        opt.attach(capture, interval=1)
        converged = bool(opt.run(fmax=numerical["fmax_eV_A"], steps=numerical["max_steps"]))
        capture(force=True)
        if not np.array_equal(atoms.cell.array, parent["cell_A"]):
            raise RuntimeError("Unexpected cell change in fixed-cell relaxation")
        fmax = float(np.linalg.norm(counter.last["forces_eV_A"], axis=1).max())
        if converged != (fmax < numerical["fmax_eV_A"]):
            raise RuntimeError("Optimizer convergence disagrees with physical force criterion")
        status = "converged" if converged else "step_limit"
    except AttemptWallLimit as exc:
        status, error = "wall_limit", str(exc)
    except (InterruptedError, KeyboardInterrupt) as exc:
        status, error, interrupted = "interrupted", str(exc), True
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
    finally:
        capture(force=True)
        last = counter.last
        arrays = dict(numbers=atoms.numbers, parent_indices=slots,
            initial_positions_A=initial_positions, cell_A=np.asarray(parent["cell_A"]),
            occupancy=occupancy, trace=np.asarray(trace, dtype=float).reshape(-1, 7),
            trajectory_positions_A=np.asarray(trajectory, float).reshape(-1, len(atoms), 3),
            snapshot_steps=np.asarray([r[0] for r in snapshots], dtype=int),
            snapshot_positions_A=np.asarray([r[1] for r in snapshots], float).reshape(-1, len(atoms), 3),
            snapshot_forces_eV_A=np.asarray([r[2] for r in snapshots], float).reshape(-1, len(atoms), 3),
            snapshot_stress_eV_A3=np.asarray([r[3] for r in snapshots], float).reshape(-1, 6),
            snapshot_energy_eV=np.asarray([r[4] for r in snapshots], float))
        if initial is not None:
            arrays.update(initial_energy_eV=np.array(initial["energy_eV"]),
                          initial_forces_eV_A=initial["forces_eV_A"], initial_stress_eV_A3=initial["stress_eV_A3"])
        if last is not None:
            arrays.update(endpoint_positions_A=last["positions_A"], endpoint_energy_eV=np.array(last["energy_eV"]),
                          endpoint_forces_eV_A=last["forces_eV_A"], endpoint_stress_eV_A3=last["stress_eV_A3"])
        atomic_npz(output / "arrays.npz", **arrays)
        mapping = None
        mapping_error = None
        if last is not None:
            try:
                mapping = mapping_diagnostics(atoms.numbers, last["positions_A"], parent, slots)
            except Exception as exc:
                mapping_error = f"{type(exc).__name__}: {exc}"
        record = dict(status=status, converged=status == "converged", task=task,
            physical_key=task["physical_key"], context_hash=context["context_hash"],
            initial_geometry_hash=initial_geometry,
            endpoint_literal_geometry_hash=geometry_hash(atoms.numbers, last["positions_A"], atoms.cell.array) if last is not None else None,
            n_atoms=len(atoms), n_target=int(occupancy.sum()), n_steps=int(opt.nsteps) if opt is not None else 0,
            fmax_target_eV_A=numerical["fmax_eV_A"], endpoint_force_max_eV_A=float(np.linalg.norm(last["forces_eV_A"], axis=1).max()) if last is not None else None,
            initial_energy_eV=initial["energy_eV"] if initial is not None else None,
            endpoint_energy_eV=last["energy_eV"] if last is not None else None,
            endpoint_labels_usable=status == "converged" and mapping is not None,
            calculator_EFS_attempts=counter.attempts, calculator_EFS_successes=counter.successes,
            calculator_EFS_failures=counter.failures, calculator_seconds=counter.seconds,
            elapsed_seconds_including_mapping=time.perf_counter() - started,
            snapshots=len(snapshots), trajectory_frames=len(trajectory), mapping=mapping,
            mapping_error=mapping_error, error=error,
            arrays_sha256=sha256(output / "arrays.npz"),
            finished_unix=time.time(), finished_UTC=datetime.now(timezone.utc).isoformat(),
            scientific_scope="Local numerical endpoint. Literal identity/site assignments do not establish a unique physical basin.",
            calls_scope="Actual logical E/F/stress calculator attempts, including initial; cached property requests and backend forwards counted separately/unknown")
        atomic_json(output / "result.json", record)
    return record, interrupted


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--max-worker-seconds", type=float, default=20000.)
    args = p.parse_args(argv)
    require_compute_allocation()
    began = time.perf_counter()
    preparation, context = check_prepared(args.prepared)
    task_record = json.loads(args.tasks.read_text())
    tasks = task_record["tasks"]
    if task_record["context_hash"] != context["context_hash"]:
        raise ValueError("Task queue context differs from frozen numerical protocol")
    parent = load_npz(args.prepared / "parent.npz")
    if sha256(args.prepared / "parent.npz") != context["parent_sha256"]:
        raise ValueError("Physical parent bytes differ")
    queue = args.queue
    for directory in ["claims", "results", "workers"]:
        (queue / directory).mkdir(parents=True, exist_ok=True)
    worker = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{socket.gethostname()}_{os.getpid()}"
    receipt_path = queue / "workers" / (worker + ".json")
    base_receipt = dict(worker=worker, worker_index=args.worker_index,
        started_UTC=datetime.now(timezone.utc).isoformat(), started_unix=time.time(),
        job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID"),
        gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), context_hash=context["context_hash"],
        task_file=dict(path=str(args.tasks.absolute()), sha256=sha256(args.tasks)),
        preparation_sha256=sha256(args.prepared / "preparation.json"),
        worker_source_sha256=sha256(__file__), common_source_sha256=sha256(Path(__file__).with_name("paired_common.py")),
        interpreter=os.sys.executable, status="loading", claimed=[], completed=[], errors=[])
    atomic_json(receipt_path, base_receipt)
    from cathode_cvae.mlip.calculators import make_calculator
    import torch
    calculator, model_receipt = make_calculator("uma", checkpoint=args.checkpoint, device=args.device, threads=args.threads)
    expected = context["teacher"]
    for key in ["head", "package_version", "dtype", "tf32", "checkpoint_hashes"]:
        if model_receipt[key] != expected[key]:
            raise ValueError("Pinned teacher context mismatch: " + key)
    if model_receipt["ase"] != context["optimizer"]["ase_version"]:
        raise ValueError("ASE version differs from frozen optimizer")
    base_receipt.update(model_receipt=model_receipt, status="running", load_and_setup_seconds=time.perf_counter() - began)
    atomic_json(receipt_path, base_receipt)
    synchronize = (lambda: torch.cuda.synchronize()) if args.device.startswith("cuda") else (lambda: None)
    def interrupted(signum, frame):
        raise InterruptedError(f"Worker signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    order = np.random.default_rng(args.worker_index + 20261108).permutation(len(tasks))
    verified_inputs = {}
    exit_code = 0
    try:
        for index in order:
            if time.perf_counter() - began >= args.max_worker_seconds:
                base_receipt["status"] = "worker_wall_limit"
                break
            task = tasks[int(index)]
            if task["context_hash"] != context["context_hash"]:
                raise ValueError("Individual task context differs")
            if physical_key(context["context_hash"], task["canonical_vacancy_key"]) != task["physical_key"]:
                raise ValueError("Task physical key differs from context/initial geometry")
            if task["task_id"] != "relax_" + task["physical_key"][:24]:
                raise ValueError("Task ID differs from physical identity")
            binding = (task["input_path"], task["input_sha256"])
            if binding not in verified_inputs:
                if sha256(task["input_path"]) != task["input_sha256"]:
                    raise ValueError("Queue input data bytes changed")
                with np.load(task["input_path"], allow_pickle=False) as archive:
                    source = archive["occupancy"] if "occupancy" in archive else 1 - archive["masks"]
                verified_inputs[binding] = source
            source = verified_inputs[binding]
            source_key, _ = canonical(source[task["input_row"]], parent["permutations"])
            if source_key != task["canonical_vacancy_key"]:
                raise ValueError("Task row does not identify its declared starting mask")
            claim_path = queue / "claims" / (task["task_id"] + ".json")
            try:
                with claim_path.open("x") as stream:
                    json.dump(dict(worker=worker, claimed_unix=time.time(), task=task,
                                   task_list_sha256=base_receipt["task_file"]["sha256"]), stream, indent=2)
            except FileExistsError:
                continue
            base_receipt["claimed"].append(task["task_id"])
            atomic_json(receipt_path, base_receipt)
            record, stop = run_attempt(calculator, parent, task, context,
                queue / "results" / task["task_id"], synchronize=synchronize)
            record["worker"] = worker
            record["model_receipt_sha256"] = identity(model_receipt)
            atomic_json(queue / "results" / task["task_id"] / "result.json", record)
            base_receipt["completed"].append(dict(task_id=task["task_id"], status=record["status"]))
            atomic_json(receipt_path, base_receipt)
            if stop:
                base_receipt["status"] = "interrupted"
                exit_code = 143
                break
        else:
            base_receipt["status"] = "finished_task_scan"
    except BaseException as exc:
        base_receipt["status"] = "worker_error"
        base_receipt["errors"].append(f"{type(exc).__name__}: {exc}")
        exit_code = 1
        raise
    finally:
        base_receipt.update(finished_UTC=datetime.now(timezone.utc).isoformat(), finished_unix=time.time(),
                            elapsed_seconds_including_loading_idle=time.perf_counter() - began)
        atomic_json(receipt_path, base_receipt)
    print(json.dumps({k: base_receipt[k] for k in ["worker", "status", "completed", "elapsed_seconds_including_loading_idle"]}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
