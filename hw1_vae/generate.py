"""Generate new images by decoding fresh standard-normal latent vectors."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .utils import load_model, resolve_device, save_grid, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hw1_vae/generated"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive.")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    rng = torch.Generator().manual_seed(args.seed)
    if (args.output_dir / "generation.json").exists() or any(args.output_dir.glob("sample_*.png")):
        raise FileExistsError(f"{args.output_dir} already contains generated images. Choose a new "
                              "--output-dir so runs with different sample counts do not mix.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The whole latent batch is drawn on CPU before chunking so changing the
    # decoder batch size preserves which latent vectors are generated.
    latents = torch.randn(args.num_samples, model.latent_dim, generator=rng)
    images = []
    with torch.inference_mode():
        for z in latents.split(args.batch_size):
            images.append(model.decode(z.to(device)).cpu())
    images = torch.cat(images)
    for index, tensor in enumerate(images):
        array = (tensor[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(args.output_dir / f"sample_{index:05d}.png")
    save_grid(images[:64], args.output_dir / "samples_grid.png")
    torch.save(latents, args.output_dir / "latents.pt")
    save_json(args.output_dir / "generation.json", {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "num_samples": args.num_samples, "seed": args.seed,
        "method": "Draw z ~ N(0,I), display decoder mean; no input image or encoder is used.",
        "pixel_range": "[0,1] mapped linearly to 8-bit grayscale PNG", "image_size": model.image_size})
    print(f"Saved {len(images)} fresh samples and grid to {args.output_dir}")


if __name__ == "__main__":
    main()
