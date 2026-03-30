import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


# Minimal wrapper around NVIDIA's train.py so experiment configs are reproducible.
def build_command(cfg: dict):
    stylegan = cfg["stylegan"]
    experiment = cfg["experiment"]

    repo_dir = Path(stylegan["repo_dir"]).resolve()
    train_py = repo_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found at: {train_py}")

    cmd = [
        "python",
        str(train_py),
        f"--outdir={experiment['outdir']}",
        f"--cfg={stylegan['cfg']}",
        f"--data={stylegan['data']}",
        f"--gpus={stylegan['gpus']}",
        f"--batch={stylegan['batch']}",
        f"--gamma={stylegan['gamma']}",
        f"--kimg={stylegan['kimg']}",
        f"--snap={stylegan['snap']}",
        f"--workers={stylegan['workers']}",
        f"--seed={stylegan['seed']}",
        f"--aug={stylegan['aug']}",
        f"--target={stylegan['target']}",
        f"--metrics={','.join(stylegan['metrics'])}",
    ]

    if stylegan.get("cond", False):
        cmd.append("--cond=1")

    if stylegan.get("mirror", False):
        cmd.append("--mirror=1")
    else:
        cmd.append("--mirror=0")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch StyleGAN2-ADA training from YAML config")
    parser.add_argument("--config", required=True, type=Path, help="Path to training YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cmd = build_command(cfg)

    print("Training command:")
    print(" ".join(shlex.quote(p) for p in cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
