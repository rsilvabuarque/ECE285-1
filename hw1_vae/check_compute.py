"""Inspect the current Python host and verify CUDA without changing its setup.

Run ``python -m hw1_vae.check_compute --test-vae --require-cuda`` on the
machine that should train. The check distinguishes missing host GPU access
from a CPU-only PyTorch installation. It never installs drivers or packages.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _pci_devices() -> list[dict[str, str]]:
    """Linux-only display/NVIDIA PCI metadata; empty on other platforms."""
    devices = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            vendor = (device / "vendor").read_text().strip()
            device_class = (device / "class").read_text().strip()
            if vendor == "0x10de" or device_class.startswith("0x03"):
                devices.append({"pci_address": device.name, "vendor": vendor,
                                "device": (device / "device").read_text().strip(),
                                "class": device_class})
        except OSError:
            continue
    return devices


def collect_report(test_vae: bool = False) -> dict:
    """Return JSON-safe evidence and, where possible, execute GPU operators."""
    executable = shutil.which("nvidia-smi")
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not executable and wsl_executable.is_file():
        executable = str(wsl_executable)
    smi = {"found": bool(executable), "ok": False, "gpus": []}
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            smi["ok"] = result.returncode == 0
            if smi["ok"]:
                smi["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            else:
                smi["error"] = (result.stderr or result.stdout).strip()[:1000]
        except (OSError, subprocess.TimeoutExpired) as error:
            smi["error"] = str(error)
    pci = _pci_devices()
    report = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "codespaces": os.environ.get("CODESPACES", "").lower() == "true",
        "wsl": "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME")),
        "nvidia_device_files": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "linux_nvidia_driver_file": Path("/proc/driver/nvidia/version").is_file(),
        "display_and_nvidia_pci_devices": pci,
        "nvidia_smi": smi,
        "torch_installed": False,
        "cuda_available": False,
        "cuda_tensor_test": {"status": "skipped"},
        "vae_test": {"status": "skipped", "requested": test_vae},
        "cuda_ready": False,
    }
    try:
        import torch
        report["torch_installed"] = True
        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_devices"] = []
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["cuda_devices"].append({
                    "index": index, "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                })
            try:
                tensor = torch.randn(16, 16, device="cuda", requires_grad=True)
                loss = (tensor @ tensor.T).square().mean()
                loss.backward()
                torch.cuda.synchronize()
                if not torch.isfinite(loss).item() or not torch.isfinite(tensor.grad).all().item():
                    raise FloatingPointError("CUDA tensor check produced nonfinite values")
                report["cuda_tensor_test"] = {"status": "passed", "device": str(tensor.device)}
                del tensor, loss
            except Exception as error:
                report["cuda_tensor_test"] = {"status": "failed", "error": str(error)}
            if test_vae and report["cuda_tensor_test"]["status"] == "passed":
                try:
                    from .model import ConvVAE, vae_loss
                    model = ConvVAE(image_size=64, latent_dim=8, base_channels=8).to("cuda")
                    images = torch.rand(2, 1, 64, 64, device="cuda")
                    reconstruction, mu, logvar = model(images)
                    loss = vae_loss(reconstruction, images, mu, logvar)["total"]
                    loss.backward()
                    torch.cuda.synchronize()
                    if reconstruction.shape != images.shape or not torch.isfinite(loss).item():
                        raise FloatingPointError("VAE CUDA check returned invalid output or loss")
                    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                               for parameter in model.parameters()):
                        raise FloatingPointError("VAE CUDA gradients are missing or nonfinite")
                    report["vae_test"] = {"status": "passed", "requested": True,
                                          "device": str(reconstruction.device),
                                          "reconstruction_shape": list(reconstruction.shape)}
                    del model, images, reconstruction, mu, logvar, loss
                except Exception as error:
                    report["vae_test"] = {"status": "failed", "requested": True, "error": str(error)}
            report["cuda_ready"] = (
                report["cuda_tensor_test"]["status"] == "passed"
                and (not test_vae or report["vae_test"]["status"] == "passed")
            )
    except ImportError as error:
        report["torch_import_error"] = str(error)
    except Exception as error:
        report["torch_probe_error"] = str(error)

    hardware_signal = (bool(report["nvidia_device_files"])
                       or report["linux_nvidia_driver_file"] or smi["ok"]
                       or any(device["vendor"] == "0x10de" for device in pci))
    report["nvidia_host_signals_present"] = hardware_signal
    if report["cuda_ready"]:
        report["conclusion"] = "CUDA is usable by this Python process; the requested execution checks passed."
    elif report["codespaces"] and not hardware_signal and not report["cuda_available"]:
        report["conclusion"] = (
            "This process is running in a cloud Codespace with no exposed NVIDIA GPU/driver signals. "
            "The laptop GPU is on a different host. Run the notebook using a Python kernel on the "
            "laptop or a GPU compute server; installing CUDA packages here cannot attach that hardware."
        )
    elif report.get("torch_cuda_build") is None and report["torch_installed"]:
        report["conclusion"] = (
            "PyTorch is a CPU-only build. On a host exposing an NVIDIA GPU and compatible driver, "
            "install a matching CUDA-enabled torch/torchvision pair, restart the kernel, and check again."
        )
    elif not report["torch_installed"]:
        report["conclusion"] = "PyTorch could not be imported. Install PyTorch in the kernel's Python environment."
    else:
        report["conclusion"] = (
            "CUDA is not ready. Inspect the driver/host visibility, PyTorch build and execution-test "
            "results above; see GPU_SETUP.md for the setup paths."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the evidence as a JSON file")
    parser.add_argument("--require-cuda", action="store_true", help="Exit nonzero unless CUDA execution succeeds")
    parser.add_argument("--test-vae", action="store_true", help="Also verify a small VAE forward/backward on CUDA")
    args = parser.parse_args()
    report = collect_report(test_vae=args.test_vae)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(serialized, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    failed = any(report[key]["status"] == "failed" for key in ("cuda_tensor_test", "vae_test"))
    if failed:
        raise SystemExit(1)
    if args.require_cuda and not report["cuda_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
