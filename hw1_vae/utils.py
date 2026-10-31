"""Small shared helpers for devices, checkpoints, and reproducible figures."""

import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import ConvVAE


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build on the "
                           "GPU host, or explicitly use --device cpu for a small check.")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible kernels where available. Exact results can still vary by
    # PyTorch release/hardware; the run records versions and device information.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment_info(device: torch.device) -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda"
            else platform.processor() or "CPU"}


def load_model(checkpoint, device) -> tuple[ConvVAE, dict]:
    # This project's checkpoints contain tensors and primitive dictionaries.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    model = ConvVAE(**{key: config[key] for key in
                       ("image_size", "latent_dim", "base_channels")})
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def save_grid(images: torch.Tensor, path, nrow: int = 8) -> None:
    """Save actual [0,1] intensities, without per-image contrast normalization."""
    images = images.detach().cpu().clamp(0, 1)
    if images.ndim != 4 or images.shape[1] != 1 or len(images) == 0:
        raise ValueError("save_grid expects a nonempty batch of grayscale images.")
    nrow = min(nrow, len(images))
    if nrow < 1:
        raise ValueError("nrow must be positive.")
    height, width = images.shape[-2:]
    padding = 2
    rows = math.ceil(len(images) / nrow)
    canvas = Image.new("L", (nrow * (width + padding) + padding,
                             rows * (height + padding) + padding), color=0)
    arrays = (images[:, 0].numpy() * 255).round().astype(np.uint8)
    for index, array in enumerate(arrays):
        canvas.paste(Image.fromarray(array),
                     (padding + (index % nrow) * (width + padding),
                      padding + (index // nrow) * (height + padding)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
