"""Conservative ASE adapters used by all portable scoring commands.

Energy is eV per simulation cell, force is eV/Angstrom, stress is eV/Angstrom^3,
ASE Voigt order xx, yy, zz, yz, xz, xy and positive in tension. A single-point
prediction keeps positions and cell unchanged; it is not an electronic SCF.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import socket
import time

import numpy as np

from .checkpoints import verify, sha256
from .registry import MODELS


def compute_guard():
    """Retain the known site's login-node restriction, without requiring Slurm elsewhere."""
    host = socket.gethostname().split(".")[0]
    nersc = os.environ.get("NERSC_HOST") or host.startswith(("perlmutter", "login")) and ".nersc.gov" in socket.getfqdn()
    if nersc and (host.startswith(("perlmutter", "login")) or not os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("On NERSC, inference/relaxation must run in a compute allocation. Request one using your site's account and queue, then rerun this command.")


def uma_predict_device(selected_device):
    """UMA accepts cpu/cuda, while its device resolver uses the current GPU."""
    import torch
    if selected_device.type == "cuda" and selected_device.index is not None:
        torch.cuda.set_device(selected_device.index)
    return selected_device.type


def make_calculator(name, checkpoint=None, device="cuda", threads=4, model_dir=None):
    """Load the pinned teacher and return (ASE calculator, provenance dict).

CUDA indices follow CUDA_VISIBLE_DEVICES; one process uses one selected GPU.
No personal account, GPU hostname or scheduler implementation is assumed.
Default FP32/TF32-off settings match the campaign rather than SDK fast defaults.
"""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name}; choose {', '.join(MODELS)}")
    compute_guard()
    import torch
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, or cuda:N (including PyTorch ROCm builds where supported upstream).")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check your allocation, driver and PyTorch wheel; CPU is available with --device cpu when supported by the backend.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, hashes = verify(name, checkpoint, model_dir)
    spec = MODELS[name]
    installed = metadata.version(spec["package"])
    if installed != spec["version"]:
        raise RuntimeError(f"{name} requires {spec['package']}=={spec['version']} for this comparison; installed {installed}. Run the project installer or select its backend environment.")
    start = time.perf_counter()
    if name == "uma":
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        settings = InferenceSettings(tf32=False, nitro=False, merge_mole=False, compile=False,
            execution_mode="general", base_precision_dtype="float32", internal_graph_gen_version=2,
            activation_checkpointing=False)
        predictor = load_predict_unit(str(checkpoint), inference_settings=settings,
                                      device=uma_predict_device(selected_device))
        actual_device = torch.device(predictor.device)
        if selected_device.index is not None and actual_device.index != selected_device.index:
            raise RuntimeError("UMA selected a different CUDA device; expose the intended GPU with CUDA_VISIBLE_DEVICES and use --device cuda")
        calculator = FAIRChemCalculator(predictor, task_name="omat")
    elif name == "mace":
        from mace.calculators import MACECalculator
        calculator = MACECalculator(model_paths=str(checkpoint), device=str(selected_device), default_dtype="float32",
            head="omat_pbe", enable_cueq=False, compile_mode=None)
    elif name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        calculator = SevenNetCalculator(model=str(checkpoint), modal="mpa", device=str(selected_device),
            enable_cueq=False, enable_flash=False, enable_oeq=False, enable_lunatp=False)
    else:
        import matgl
        matgl.set_backend("PyG")
        matgl.set_default_dtype("float", 32)
        from matgl.ext.ase import PESCalculator
        potential = matgl.load_model(str(checkpoint)).to(device=selected_device, dtype=torch.float32).eval()
        if not potential.calc_forces or not potential.calc_stresses:
            raise ValueError("The loaded MatGL potential does not provide both force and stress.")
        calculator = PESCalculator(potential=potential, stress_unit="eV/A3", stress_weight=1.0, use_voigt=True)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    return calculator, {
        "model": name, "checkpoint_name": spec["name"], "head": spec["head"],
        "adapter_source_sha256": sha256(__file__),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_hashes": hashes,
        "package": spec["package"], "package_version": installed, "torch": torch.__version__,
        "ase": metadata.version("ase"), "dtype": "float32", "tf32": False,
        "nitro": False, "compile": False, "dispersion": False, "device": str(selected_device),
        "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
        "load_seconds": time.perf_counter() - start, "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(), "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stress_convention": "ASE tensile-positive; xx yy zz yz xz xy; eV/A^3",
    }


def evaluate(calculator, atoms):
    atoms.calc = calculator
    calculator.reset()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(), dtype=float)
    if forces.shape != (len(atoms), 3) or stress.shape != (6,):
        raise ValueError("Unexpected force/stress shape")
    if not all(np.isfinite(value).all() for value in [energy, forces, stress]):
        raise ValueError("Non-finite energy/force/stress")
    return energy, forces, stress
