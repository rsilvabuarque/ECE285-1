"""Load the homework's chest X-ray images without using the diagnostic labels.

The VAE learns from images alone. NORMAL and PNEUMONIA are both included;
their directory names are useful for checking that the dataset is complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"


def resolve_data_root(data_dir: str | Path) -> Path:
    """Find one root containing train/val/test, including common Kaggle wrappers.

    Some extracts contain a second copy under chest_xray/chest_xray. Selecting
    the first complete root, then reading only its exact split folders, avoids
    accidentally counting both copies or mixing training and test images.
    """
    supplied = Path(data_dir).expanduser().resolve()
    candidates = (supplied, supplied / "chest_xray", supplied / "chest_xray" / "chest_xray")
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate
    checked = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a chest X-ray dataset root containing train/, val/, and test/.\n"
        f"Checked:\n{checked}\n"
        "Pass --data-dir pointing at the extracted chest_xray directory, or run\n"
        "python -m hw1_vae.data --download to explicitly download the Kaggle dataset."
    )


def _is_image(path: Path, class_dir: Path) -> bool:
    relative_parts = path.relative_to(class_dir).parts
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX" for part in relative_parts)
    )


class ChestXrayDataset(Dataset):
    """One official split, resized to grayscale tensors of shape [1, H, W].

    Preprocessing is deterministic for every split: convert to grayscale,
    bilinear-resize to a square, and scale uint8 intensities to [0, 1]. No
    augmentation or label conditioning changes the reconstruction target.
    """

    def __init__(self, data_dir: str | Path, split: str, image_size: int = 64) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; received {split!r}")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.root = resolve_data_root(data_dir)
        self.split = split
        self.image_size = image_size
        self.class_counts: dict[str, int] = {}
        image_paths: list[Path] = []
        for class_name in CLASSES:
            class_dir = self.root / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing dataset class directory: {class_dir}. "
                    "Each official split must contain NORMAL/ and PNEUMONIA/."
                )
            paths = sorted(path for path in class_dir.rglob("*") if _is_image(path, class_dir))
            if not paths:
                raise FileNotFoundError(f"No supported image files found in {class_dir}")
            self.class_counts[class_name] = len(paths)
            image_paths.extend(paths)
        self.image_paths = tuple(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("L").resize(
                    (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
                )
                pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read chest X-ray image {path}: {error}") from error
        return torch.from_numpy(pixels).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Existing dataset root or its extraction parent")
    source.add_argument(
        "--download", action="store_true", help="Explicitly download the homework dataset with kagglehub"
    )
    args = parser.parse_args()
    if args.download:
        try:
            import kagglehub
        except ImportError:
            parser.error("--download requires kagglehub: python -m pip install kagglehub")
        data_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    else:
        data_dir = args.data_dir
    root = resolve_data_root(data_dir)
    print(f"Resolved dataset root: {root}")
    print(f"{'split':<8} {'NORMAL':>9} {'PNEUMONIA':>12} {'total':>9}")
    for split in SPLITS:
        dataset = ChestXrayDataset(root, split)
        print(
            f"{split:<8} {dataset.class_counts['NORMAL']:>9} "
            f"{dataset.class_counts['PNEUMONIA']:>12} {len(dataset):>9}"
        )
    print("Preprocessing: grayscale, bilinear resize to 64x64, float32 in [0, 1].")
    print("Use this resolved root as --data-dir for training and evaluation.")


if __name__ == "__main__":
    main()
