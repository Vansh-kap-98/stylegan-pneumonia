#!/usr/bin/env bash
set -euo pipefail

# Clones NVIDIA StyleGAN2-ADA-PyTorch into third_party.
# Pin to a stable commit for reproducibility.

REPO_URL="https://github.com/NVlabs/stylegan2-ada-pytorch.git"
DEST_DIR="third_party/stylegan2-ada-pytorch"

if [ -d "$DEST_DIR/.git" ]; then
  echo "Repo already exists at $DEST_DIR"
  exit 0
fi

mkdir -p third_party
git clone "$REPO_URL" "$DEST_DIR"

echo "Cloned $REPO_URL to $DEST_DIR"

echo "Optional: pin commit with"
echo "  cd $DEST_DIR && git checkout <commit-hash>"
