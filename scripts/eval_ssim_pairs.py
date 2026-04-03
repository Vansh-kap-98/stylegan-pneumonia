#!/usr/bin/env python3
"""Compute class-wise SSIM between synthetic and real chest X-ray images.

Usage example:
python scripts/eval_ssim_pairs.py \
  --real-zip datasets/pneumonia_256_conditional.zip \
  --synthetic-normal-dir outputs/snapshot200/class0 \
  --synthetic-pneumonia-dir outputs/snapshot200/class1 \
  --max-pairs 256
"""

import argparse
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSIM against class-matched real images")
    parser.add_argument("--real-zip", required=True, help="Path to StyleGAN dataset zip with dataset.json labels")
    parser.add_argument("--synthetic-normal-dir", required=True, help="Dir containing synthetic Normal images")
    parser.add_argument("--synthetic-pneumonia-dir", required=True, help="Dir containing synthetic Pneumonia images")
    parser.add_argument("--max-pairs", type=int, default=256, help="Max pairs per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_image_grayscale_from_bytes(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.float32)


def load_real_by_class(real_zip_path: Path) -> Dict[int, List[np.ndarray]]:
    real_by_class: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    with zipfile.ZipFile(real_zip_path, "r") as zf:
        with zf.open("dataset.json") as f:
            meta = json.load(f)

        labels = meta.get("labels")
        if labels is None:
            raise ValueError("dataset.json does not contain labels")

        for rel_path, label in labels:
            if isinstance(label, list):
                # Defensive handling for one-hot labels.
                label = int(np.argmax(label))
            label = int(label)
            if label not in (0, 1):
                continue
            with zf.open(rel_path) as img_file:
                real_by_class[label].append(load_image_grayscale_from_bytes(img_file.read()))

    if not real_by_class[0] or not real_by_class[1]:
        raise ValueError("Could not find both class 0 and class 1 images in real dataset zip")

    return real_by_class


def load_synthetic_images(dir_path: Path) -> List[np.ndarray]:
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted([p for p in dir_path.glob("**/*") if p.suffix.lower() in allowed_ext])
    if not paths:
        raise ValueError(f"No synthetic images found in {dir_path}")

    images = []
    for path in paths:
        img = Image.open(path).convert("L")
        images.append(np.asarray(img, dtype=np.float32))
    return images


def class_ssim_scores(
    synth_images: List[np.ndarray],
    real_images: List[np.ndarray],
    max_pairs: int,
    rng: random.Random,
) -> List[float]:
    n = min(max_pairs, len(synth_images), len(real_images))
    synth_subset = rng.sample(synth_images, n)
    real_subset = rng.sample(real_images, n)

    scores = []
    for synth, real in zip(synth_subset, real_subset):
        # Ensure same shape for SSIM.
        if synth.shape != real.shape:
            real_img = Image.fromarray(real.astype(np.uint8)).resize((synth.shape[1], synth.shape[0]), Image.BILINEAR)
            real = np.asarray(real_img, dtype=np.float32)

        score = ssim(synth, real, data_range=255.0)
        scores.append(float(score))

    return scores


def summarize(scores: List[float]) -> Tuple[float, float]:
    arr = np.asarray(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    real_by_class = load_real_by_class(Path(args.real_zip))
    synth_normal = load_synthetic_images(Path(args.synthetic_normal_dir))
    synth_pneu = load_synthetic_images(Path(args.synthetic_pneumonia_dir))

    normal_scores = class_ssim_scores(synth_normal, real_by_class[0], args.max_pairs, rng)
    pneu_scores = class_ssim_scores(synth_pneu, real_by_class[1], args.max_pairs, rng)
    overall_scores = normal_scores + pneu_scores

    n_mean, n_std = summarize(normal_scores)
    p_mean, p_std = summarize(pneu_scores)
    o_mean, o_std = summarize(overall_scores)

    print("SSIM evaluation complete")
    print(f"Normal class    : mean={n_mean:.4f}, std={n_std:.4f}, n={len(normal_scores)}")
    print(f"Pneumonia class : mean={p_mean:.4f}, std={p_std:.4f}, n={len(pneu_scores)}")
    print(f"Overall         : mean={o_mean:.4f}, std={o_std:.4f}, n={len(overall_scores)}")


if __name__ == "__main__":
    main()
