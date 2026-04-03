#!/usr/bin/env python3
"""Create blinded packet CSVs for radiologist visual Turing test.

Generates:
- blinded_review.csv: randomized image list without source labels
- answer_key.csv: mapping from image_id to true source/class (keep hidden from reviewers)
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build blinded visual Turing packet")
    parser.add_argument("--real-normal-dir", required=True)
    parser.add_argument("--real-pneumonia-dir", required=True)
    parser.add_argument("--synth-normal-dir", required=True)
    parser.add_argument("--synth-pneumonia-dir", required=True)
    parser.add_argument("--n-per-group", type=int, default=100, help="Images sampled from each of 4 groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/visual_turing")
    return parser.parse_args()


def collect_images(dir_path: Path) -> List[Path]:
    images = [p for p in dir_path.glob("**/*") if p.suffix.lower() in ALLOWED_EXT]
    if not images:
        raise ValueError(f"No images found in {dir_path}")
    return sorted(images)


def sample_group(paths: List[Path], n: int, rng: random.Random) -> List[Path]:
    if len(paths) < n:
        raise ValueError(f"Not enough images: requested {n}, found {len(paths)}")
    return rng.sample(paths, n)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    groups: List[Tuple[str, str, List[Path]]] = [
        ("real", "normal", collect_images(Path(args.real_normal_dir))),
        ("real", "pneumonia", collect_images(Path(args.real_pneumonia_dir))),
        ("synthetic", "normal", collect_images(Path(args.synth_normal_dir))),
        ("synthetic", "pneumonia", collect_images(Path(args.synth_pneumonia_dir))),
    ]

    rows = []
    idx = 1
    for source, cls, paths in groups:
        sampled = sample_group(paths, args.n_per_group, rng)
        for p in sampled:
            rows.append(
                {
                    "image_id": f"img_{idx:04d}",
                    "image_path": str(p).replace("\\", "/"),
                    "source": source,
                    "class": cls,
                }
            )
            idx += 1

    rng.shuffle(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blinded_path = outdir / "blinded_review.csv"
    key_path = outdir / "answer_key.csv"

    with blinded_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "reviewer_label_real_or_synthetic", "confidence_1to5"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "reviewer_label_real_or_synthetic": "",
                    "confidence_1to5": "",
                }
            )

    with key_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "source", "class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {blinded_path}")
    print(f"Created: {key_path}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()
