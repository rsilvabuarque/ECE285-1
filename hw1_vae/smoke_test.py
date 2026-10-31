"""Small CPU checks for the VAE pipeline, with no datasets or weights downloaded.

Run from the repository root with ``python -m hw1_vae.smoke_test``.
Temporary fixture images are artificial; passing these checks is not evidence
that a model has learned useful chest X-ray structure.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import CLASSES, SPLITS, ChestXrayDataset, resolve_data_root
from .evaluate import compute_feature_metrics, preprocess_inception
from .model import ConvVAE, vae_loss
from .utils import load_model


def check_data(temp_dir: Path) -> None:
    root = temp_dir / "extracted" / "chest_xray"
    pixels = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            Image.fromarray(pixels).save(class_dir / "example.jpeg")
            # Invalid metadata files must never become training examples.
            (class_dir / "._example.jpeg").write_text("AppleDouble metadata")
            hidden_dir = class_dir / "__MACOSX"
            hidden_dir.mkdir()
            (hidden_dir / "ignored.jpeg").write_text("not an image")
        nested = root / split / "NORMAL" / "patient_subfolder"
        nested.mkdir()
        Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(nested / "rgb.png")
        # An extraction can also contain a nested duplicate dataset. It must
        # not be traversed when collecting images from the selected root.
        for class_name in CLASSES:
            duplicate = root / "chest_xray" / split / class_name
            duplicate.mkdir(parents=True)
            Image.fromarray(pixels).save(duplicate / "duplicate.jpeg")
    assert resolve_data_root(root.parent) == root
    assert resolve_data_root(root / "chest_xray") == root / "chest_xray"
    for split in SPLITS:
        dataset = ChestXrayDataset(root.parent, split, image_size=32)
        assert len(dataset) == 3
        assert dataset.class_counts == {"NORMAL": 2, "PNEUMONIA": 1}
        for path in dataset.image_paths:
            assert path.is_relative_to(root / split)
        image = dataset[0]
        assert image.shape == (1, 32, 32)
        assert image.dtype == torch.float32
        assert torch.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
        torch.testing.assert_close(image, dataset[0], rtol=0, atol=0)
    try:
        ChestXrayDataset(temp_dir / "does_not_exist", "train")
    except FileNotFoundError as error:
        assert "train/" in str(error) and "--data-dir" in str(error)
    else:
        raise AssertionError("Missing datasets must fail with a useful error")
    try:
        ChestXrayDataset(root, "training")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown split names must fail instead of mixing data")
    (root / "test" / "PNEUMONIA" / "example.jpeg").unlink()
    try:
        ChestXrayDataset(root, "test")
    except FileNotFoundError as error:
        assert "No supported image files" in str(error) and "PNEUMONIA" in str(error)
    else:
        raise AssertionError("Both diagnostic directories must contain images")
    print("PASS data: official splits, nested paths, duplicate/metadata exclusion, grayscale scaling")


def check_model_and_loss(temp_dir: Path) -> None:
    torch.manual_seed(42)
    config = {"image_size": 32, "latent_dim": 8, "base_channels": 8}
    model = ConvVAE(**config)
    for batch_size in (1, 4):
        model.zero_grad(set_to_none=True)
        images = torch.rand(batch_size, 1, config["image_size"], config["image_size"])
        reconstructed, mu, logvar = model(images)
        assert reconstructed.shape == images.shape
        assert mu.shape == logvar.shape == (batch_size, config["latent_dim"])
        assert torch.isfinite(reconstructed).all()
        assert 0 <= reconstructed.min() <= reconstructed.max() <= 1
        losses = vae_loss(reconstructed, images, mu, logvar)
        assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
        losses["total"].backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        for head in (model.fc_mu, model.fc_logvar):
            assert head.weight.grad.abs().sum() > 0, "Both posterior heads need a gradient"

    zeros = torch.zeros(4, config["latent_dim"])
    image = torch.rand(4, 1, 32, 32)
    prior_losses = vae_loss(image, image, zeros, zeros)
    torch.testing.assert_close(prior_losses["kl"], torch.tensor(0.0))
    torch.testing.assert_close(prior_losses["reconstruction"], torch.tensor(0.0))
    shifted_losses = vae_loss(image, image, torch.ones_like(zeros), zeros)
    torch.testing.assert_close(shifted_losses["kl"], torch.tensor(config["latent_dim"] / 2))
    random_losses = vae_loss(image * 0.8, image, torch.randn_like(zeros), torch.randn_like(zeros))
    assert random_losses["kl"] >= 0
    torch.testing.assert_close(
        random_losses["total"], random_losses["reconstruction"] + 0.001 * random_losses["kl"]
    )

    # Averaging across the batch must preserve the objective when identical
    # examples are repeated; this catches accidental batch-summed KL.
    one_mu, one_logvar = torch.randn(1, 8), torch.randn(1, 8)
    one_target, one_prediction = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    single = vae_loss(one_prediction, one_target, one_mu, one_logvar)
    repeated = vae_loss(
        one_prediction.repeat(4, 1, 1, 1), one_target.repeat(4, 1, 1, 1),
        one_mu.repeat(4, 1), one_logvar.repeat(4, 1),
    )
    for key in single:
        torch.testing.assert_close(single[key], repeated[key])

    checkpoint_path = temp_dir / "untrained_smoke_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": config, "epoch": 0}, checkpoint_path)
    restored, state = load_model(checkpoint_path, torch.device("cpu"))
    assert state["epoch"] == 0 and not restored.training
    latent = torch.randn(3, config["latent_dim"])
    model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model.decode(latent), restored.decode(latent), rtol=0, atol=0)
        full_size = ConvVAE(image_size=64, latent_dim=8, base_channels=8)
        output, _, _ = full_size(torch.rand(1, 1, 64, 64))
        assert output.shape == (1, 1, 64, 64)
    print("PASS model: batch sizes 1/4, 32/64px output, finite gradients, KL formula, checkpoint reload")


def check_metrics() -> None:
    # These tests use synthetic probabilities and an identity extractor. They
    # check metric plumbing without constructing Inception or fetching weights.
    from scipy.linalg import LinAlgWarning

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    with warnings.catch_warnings():
        # This analytic two-class fixture necessarily has singular covariance.
        # Suppress that specific expected warning only for this offline check.
        warnings.filterwarnings("ignore", message="Matrix is singular.*", category=LinAlgWarning)
        metrics = compute_feature_metrics(features, features)
    assert abs(metrics["fid"]) < 1e-6
    assert abs(metrics["inception_score"] - 2.0) < 1e-6
    processed = preprocess_inception(torch.zeros(2, 1, 32, 32))
    assert processed.shape == (2, 3, 299, 299) and torch.isfinite(processed).all()
    expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(processed[0, :, 0, 0], expected)
    print("PASS metrics: identical-set FID=0, known Inception Score=2, shared image preprocessing")


def main() -> None:
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="ece285-vae-smoke-") as directory:
        temp_dir = Path(directory)
        check_data(temp_dir)
        check_model_and_loss(temp_dir)
        check_metrics()
    print("All offline CPU checks passed. No dataset or pretrained weights were downloaded.")


if __name__ == "__main__":
    main()
