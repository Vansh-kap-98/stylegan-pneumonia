import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.suffix.lower() in VALID_EXTS:
            yield path


def detect_label(path: Path) -> int:
    parts = [p.upper() for p in path.parts]
    if "NORMAL" in parts:
        return LABEL_MAP["NORMAL"]
    if "PNEUMONIA" in parts:
        return LABEL_MAP["PNEUMONIA"]
    raise ValueError(f"Unable to infer label from path: {path}")


def ensure_rgb_256(src: Path, dst: Path, image_size: int) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="PNG", compress_level=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mendeley chest X-ray dataset for StyleGAN2-ADA")
    parser.add_argument("--input-root", required=True, type=Path, help="Root folder containing NORMAL and PNEUMONIA")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder for resized images")
    parser.add_argument("--image-size", type=int, default=256, help="Target square resolution")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    out_normal = output_root / "0_normal"
    out_pneumonia = output_root / "1_pneumonia"
    out_normal.mkdir(parents=True, exist_ok=True)
    out_pneumonia.mkdir(parents=True, exist_ok=True)

    all_images = list(iter_images(input_root))
    if args.limit > 0:
        all_images = all_images[: args.limit]

    records = []
    counters = {0: 0, 1: 0}

    for src in tqdm(all_images, desc="Preparing images"):
        label = detect_label(src)
        idx = counters[label]
        counters[label] += 1

        if label == 0:
            dst = out_normal / f"normal_{idx:06d}.png"
        else:
            dst = out_pneumonia / f"pneumonia_{idx:06d}.png"

        ensure_rgb_256(src, dst, args.image_size)
        records.append({"path": str(dst.relative_to(output_root)).replace("\\", "/"), "label": label})

    labels_payload = {
        "labels": [[rec["path"], rec["label"]] for rec in records]
    }

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_size": args.image_size,
        "total": len(records),
        "class_counts": {"0_normal": counters[0], "1_pneumonia": counters[1]},
        "labels": LABEL_MAP,
        "records": records,
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_json_path = output_root / "dataset.json"
    dataset_json_path.write_text(json.dumps(labels_payload), encoding="utf-8")

    print(f"Prepared {len(records)} images at {output_root}")
    print(f"Class counts: normal={counters[0]} pneumonia={counters[1]}")
    print(f"Manifest: {manifest_path}")
    print(f"StyleGAN labels: {dataset_json_path}")
    print("Next: run StyleGAN dataset_tool.py to create the zip dataset.")


if __name__ == "__main__":
    main()
