"""Train on Kaggle's train split; select checkpoints using its val split."""

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ChestXrayDataset
from .model import ConvVAE, vae_loss
from .utils import environment_info, resolve_device, save_grid, save_json, seed_everything


def run_epoch(model, loader, device, kl_weight, optimizer=None):
    """Average per-image losses, weighting a short final batch correctly."""
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for images in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(images)
            losses = vae_loss(reconstruction, images, mu, logvar, kl_weight)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Nonfinite loss. Check input scaling and learning rate.")
            if training:
                losses["total"].backward()
                optimizer.step()
            for name, value in losses.items():
                sums[name] += value.item() * len(images)
            count += len(images)
    if not count:
        raise ValueError("The data loader is empty.")
    return {name: total / count for name, total in sums.items()}


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for axis, key, label in zip(axes, ("total", "reconstruction", "kl"),
                                ("Weighted objective", "Pixel MSE", "KL (nats/image)")):
        for split in ("train", "val"):
            axis.plot([row["epoch"] for row in history],
                      [row[f"{split}_{key}"] for row in history], label=split)
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def data_manifest(datasets):
    """Track which relative filenames and file sizes belong to each split."""
    manifest = {}
    for split, dataset in datasets.items():
        manifest[split] = [
            {"file": path.relative_to(dataset.root).as_posix(), "bytes": path.stat().st_size}
            for path in dataset.image_paths
        ]
    # Not an image-content hash: it catches accidental changes of split/layout,
    # while allowing a run to resume after moving the data to another computer.
    import json
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, digest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hw1_vae"))
    parser.add_argument("--epochs", type=int, default=50, help="Total target epochs, including resumed epochs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--kl-warmup-epochs", type=int, default=0,
                        help="Optional linear ramp of training KL weight; validation always uses target weight.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Resume last.pt at the next epoch; saved hyperparameters win.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.num_workers < 0:
        raise ValueError("epochs must be positive and num-workers nonnegative.")
    device = resolve_device(args.device)
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    if resume:
        for key in ("batch_size", "image_size", "latent_dim", "base_channels",
                    "learning_rate", "kl_weight", "kl_warmup_epochs", "seed"):
            config[key] = resume["config"][key]
    if (config["batch_size"] < 1 or config["learning_rate"] <= 0
            or config["kl_weight"] < 0 or config["kl_warmup_epochs"] < 0):
        raise ValueError("Invalid batch size, learning rate, or KL settings.")
    seed_everything(config["seed"])
    datasets = {split: ChestXrayDataset(args.data_dir, split, config["image_size"])
                for split in ("train", "val")}
    manifest, manifest_digest = data_manifest(datasets)
    if resume and resume["manifest_digest"] != manifest_digest:
        raise ValueError("Dataset filenames/sizes differ from the saved run; start a separate output directory.")
    output = args.output_dir
    occupied = any((output / name).exists() for name in ("last.pt", "best.pt", "history.csv"))
    if not resume and occupied:
        raise FileExistsError(f"{output} already contains a run. Use --resume or a new --output-dir.")
    if resume and occupied and output.resolve() != args.resume.parent.resolve():
        raise FileExistsError(f"{output} already contains a different run. Resume in the checkpoint's "
                              "directory or choose an empty --output-dir.")
    if resume and args.epochs <= resume["epoch"]:
        raise ValueError(f"Checkpoint is already at epoch {resume['epoch']}; --epochs must be larger.")
    previous_best = None
    if resume:
        candidate = args.resume.parent / "best.pt"
        if candidate.is_file():
            best_state = torch.load(candidate, map_location="cpu", weights_only=True)
            if (best_state["manifest_digest"] != resume["manifest_digest"]
                    or best_state["best_val"] != resume["best_val"]
                    or best_state["epoch"] > resume["epoch"]):
                raise ValueError("The sibling best.pt does not match the resumed checkpoint's "
                                 "selection history. Use the latest last.pt with its matching best.pt.")
            previous_best = candidate
        elif resume["history"][-1]["val_total"] == resume["best_val"]:
            previous_best = args.resume
        else:
            raise FileNotFoundError("The historical best checkpoint is missing. Keep best.pt next "
                                    "to last.pt when moving a run to another computer.")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", environment_info(device))
    save_json(output / "data_manifest.json", manifest)
    print(f"Device: {device}; split sizes: "
          f"train={len(datasets['train'])}, val={len(datasets['val'])}", flush=True)
    if len(datasets["val"]) < 100:
        print("Validation split is small: checkpoint selection can be noisy. Keep test data held out.", flush=True)

    train_generator = torch.Generator().manual_seed(config["seed"])
    loaders = {
        split: DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train",
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          generator=train_generator if split == "train" else None, drop_last=False)
        for split, dataset in datasets.items()
    }
    model = ConvVAE(**{key: config[key] for key in ("image_size", "latent_dim", "base_channels")}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    history, best_val, start_epoch = [], math.inf, 1
    if resume:
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        history, best_val, start_epoch = resume["history"], resume["best_val"], resume["epoch"] + 1
        train_generator.set_state(resume["train_generator_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_states") is not None:
            if len(resume["cuda_rng_states"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
            else:
                print("CUDA device count changed; continuing with newly seeded GPU random streams. "
                      "Training remains resumable, but its stochastic trajectory will change.", flush=True)
                torch.cuda.manual_seed_all(config["seed"] + start_epoch)
        # When resuming into a new directory, carry over the previously selected
        # best checkpoint if it is available next to the resumed checkpoint.
        if previous_best.resolve() != (output / "best.pt").resolve():
            import shutil
            shutil.copy2(previous_best, output / "best.pt")

    fixed_rng = torch.Generator().manual_seed(config["seed"] + 1)
    fixed_z = torch.randn(64, config["latent_dim"], generator=fixed_rng).to(device)
    preview_images = torch.stack([datasets["val"][i] for i in range(min(8, len(datasets["val"])))]).to(device)
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.perf_counter()
        weight = config["kl_weight"]
        if config["kl_warmup_epochs"]:
            weight *= min(1.0, epoch / config["kl_warmup_epochs"])
        train_metrics = run_epoch(model, loaders["train"], device, weight, optimizer)
        # Reuse validation randomness every epoch and preserve the training RNG.
        # This makes changes in val loss easier to interpret on the tiny split.
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config["seed"] + 1000)
            val_metrics = run_epoch(model, loaders["val"], device, config["kl_weight"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start_time
        row = {"epoch": epoch, "seconds": duration, "train_kl_weight": weight,
               "val_kl_weight": config["kl_weight"],
               **{f"train_{key}": value for key, value in train_metrics.items()},
               **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "config": config, "epoch": epoch, "best_val": best_val, "history": history,
                 "manifest_digest": manifest_digest,
                 "torch_rng_state": torch.get_rng_state(),
                 "train_generator_state": train_generator.get_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        # Write each checkpoint atomically so an interrupted write cannot replace
        # the last usable checkpoint with a partial file.
        for name in (["last.pt", "best.pt"] if is_best else ["last.pt"]):
            temporary = output / (name + ".tmp")
            torch.save(state, temporary)
            temporary.replace(output / name)
        with torch.inference_mode():
            model.eval()
            save_grid(model.decode(fixed_z), output / "previews" / f"samples_{epoch:03d}.png")
            mu, _ = model.encode(preview_images)
            reconstructed = model.decode(mu)
            pairs = torch.stack((preview_images, reconstructed), dim=1).flatten(0, 1)
            save_grid(pairs, output / "reconstructions.png", nrow=8)
        plot_history(history, output / "loss_curves.png")
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_metrics['total']:.6f}, "
              f"val={val_metrics['total']:.6f}, MSE={val_metrics['reconstruction']:.6f}, "
              f"KL={val_metrics['kl']:.3f}, {duration:.1f}s" + (" [best]" if is_best else ""), flush=True)
        print(f"Approximate remaining training/validation time at this rate: "
              f"{duration * (args.epochs - epoch) / 60:.1f} min (excludes plots/checkpoint writes).", flush=True)
    print(f"Finished. Selected model: {output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
