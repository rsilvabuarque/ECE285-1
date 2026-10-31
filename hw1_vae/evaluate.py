"""Evaluate fresh prior samples with the homework's PyTorch-Ignite metrics.

This deliberately uses 1000 ImageNet class probabilities, as the default
extractor in the linked tutorial does. It is NOT the usual 2048-pool-feature
FID. Both metrics share one frozen pretrained Inception network.

Sources:
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
https://docs.pytorch.org/ignite/generated/ignite.metrics.FID.html
https://github.com/pytorch/ignite/blob/v0.5.3/ignite/metrics/gan/utils.py

Unlike the tutorial's GAN [-1,1] tensors, this VAE produces [0,1] images.
Preprocessing is explicit and identical for real and generated images:
repeat grayscale into RGB, bilinear resize to 299 with antialiasing, then
ImageNet normalization. These choices must accompany reported scores.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import ChestXrayDataset
from .utils import load_model, resolve_device, save_json


def preprocess_inception(images: torch.Tensor) -> torch.Tensor:
    """Map a batch N x 1 x H x W in [0,1] to normalized RGB299."""
    if images.ndim != 4 or images.shape[1] != 1 or images.shape[0] == 0:
        raise ValueError("Expected a nonempty N x 1 x H x W grayscale batch.")
    if not torch.isfinite(images).all():
        raise ValueError("Inception input contains nonfinite pixels.")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Expected pixels in [0,1] before Inception preprocessing.")
    rgb = images.float().repeat(1, 3, 1, 1)
    rgb = F.interpolate(
        rgb, size=(299, 299), mode="bilinear", align_corners=False, antialias=True
    )
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


class InceptionProbabilities(nn.Module):
    """Frozen ImageNet Inception-v3; constructing it may download weights."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        # torchvision's transform_input adapts ImageNet-normalized input to
        # the normalization expected by the original Inception weights.
        self.network = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True
        )
        self.network.requires_grad_(False)
        self.network.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(preprocess_inception(images))
        return torch.softmax(logits, dim=1)


def compute_feature_metrics(
    real_probabilities: torch.Tensor, generated_probabilities: torch.Tensor
) -> dict[str, float]:
    """Use Ignite on cached features; also supports tiny offline unit checks.

    Inputs are equal-size N x K probability matrices. Identity extractors
    avoid a second Inception forward and make the feature definition explicit.
    Ignite accumulates its moments in float64 and uses SciPy's matrix root.
    """
    from ignite.metrics import FID, InceptionScore

    if (
        real_probabilities.ndim != 2
        or generated_probabilities.shape != real_probabilities.shape
        or real_probabilities.shape[0] < 2
        or real_probabilities.shape[1] < 2
    ):
        raise ValueError("Expected matching N x K matrices with N >= 2 and K >= 2.")
    for probabilities in (real_probabilities, generated_probabilities):
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Class probabilities must be finite and nonnegative.")
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities[:, 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Each probability row must sum to one.")

    # Accumulation on CPU avoids reserving GPU space for covariance matrices.
    real = real_probabilities.detach().cpu().double()
    generated = generated_probabilities.detach().cpu().double()
    dimensions = real.shape[1]
    fid_metric = FID(num_features=dimensions, feature_extractor=nn.Identity())
    generated_is = InceptionScore(num_features=dimensions, feature_extractor=nn.Identity())
    real_is = InceptionScore(num_features=dimensions, feature_extractor=nn.Identity())
    fid_metric.update((generated, real))
    generated_is.update(generated)
    real_is.update(real)
    raw_fid = float(fid_metric.compute())
    score = float(generated_is.compute())
    reference_score = float(real_is.compute())
    if not all(math.isfinite(value) for value in (raw_fid, score, reference_score)):
        raise FloatingPointError("Metric computation returned a nonfinite result.")
    if raw_fid < -1e-6:
        raise FloatingPointError(f"FID should be nonnegative; numerical result was {raw_fid}.")
    return {
        "fid": max(0.0, raw_fid),
        "fid_raw_before_roundoff_clamp": raw_fid,
        "inception_score": score,
        "real_inception_score_reference": reference_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Number of real/fresh generated images; 0 uses every official test image.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 0 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("num-samples/workers must be nonnegative; batch-size must be positive.")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    test = ChestXrayDataset(args.data_dir, split="test", image_size=model.image_size)
    count = len(test) if args.num_samples == 0 else args.num_samples
    if not 2 <= count <= len(test):
        raise ValueError(f"num-samples must be between 2 and {len(test)}, or 0 for all test images.")

    # Separate streams keep test selection independent from prior draws.
    selection_rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(test), generator=selection_rng)[:count].tolist()
    prior_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    # Draw once before chunking, so decoder batch size cannot change which
    # latent vectors belong to this seeded evaluation.
    latents = torch.randn(count, model.latent_dim, generator=prior_rng)
    loader = DataLoader(
        Subset(test, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    print("Loading frozen ImageNet Inception-v3 (first use downloads approximately 104 MB).", flush=True)
    extractor = InceptionProbabilities().to(device).eval()
    real_features: list[torch.Tensor] = []
    generated_features: list[torch.Tensor] = []
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            latent = latents[seen:seen + images.shape[0]].to(device)
            generated = model.decode(latent)
            real_features.append(extractor(images).cpu())
            generated_features.append(extractor(generated).cpu())
            seen += images.shape[0]
            print(f"Extracted features for {seen}/{count} real and generated images.", flush=True)
    # The network is no longer needed while SciPy computes the covariance root.
    del extractor
    values = compute_feature_metrics(torch.cat(real_features), torch.cat(generated_features))
    warnings = [
        "ImageNet features and labels differ from the chest X-ray domain; these scores do not establish medical validity.",
        "FID depends on sample count, preprocessing, weights and feature layer; compare only identical protocols.",
        "This is 1000-probability-feature FID following the Ignite tutorial, not standard 2048-pool-feature FID.",
    ]
    if count <= 1000:
        warnings.append(
            "There are fewer samples than feature dimensions, so empirical covariances are rank deficient and the finite-sample FID estimate is sensitive to sampling."
        )
    result = {
        **values,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_dir": str(args.data_dir.resolve()),
        "reference_split": "test",
        "num_real_images": count,
        "num_generated_images": count,
        "batch_size": args.batch_size,
        "device": str(device),
        "model_image_size": model.image_size,
        "latent_dim": model.latent_dim,
        "reference_indices": indices,
        "seed": args.seed,
        "prior_seed": args.seed + 1,
        "generation": "z drawn independently from N(0,I); decoder means; no encoder/reconstruction input",
        "fid_feature_type": "1000 ImageNet class probabilities",
        "inception_weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
        "preprocessing": {
            "source_pixels": "grayscale [0,1] at model resolution, for both real and generated",
            "rgb": "repeat grayscale channel three times",
            "resize": "torch interpolate to 299x299, bilinear, align_corners=False, antialias=True",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "torchvision_transform_input": True,
        },
        "directions": {"fid": "lower is better", "inception_score": "higher is conventionally better"},
        "inception_score_protocol": "one aggregate score over all generated samples, not mean/std of split scores",
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "pytorch-ignite", "numpy", "scipy")
        },
        "limitations": warnings,
        "source": "https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "metrics.json"
    save_json(output, result)
    print(f"FID (1000 probability features): {values['fid']:.6f}")
    print(f"Inception Score: {values['inception_score']:.6f}")
    print(f"Saved protocol and scores to {output}")


if __name__ == "__main__":
    main()
