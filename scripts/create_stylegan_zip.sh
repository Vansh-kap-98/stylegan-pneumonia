#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/create_stylegan_zip.sh <stylegan_repo_dir> <source_dir> <dest_zip>
# Example:
#   ./scripts/create_stylegan_zip.sh third_party/stylegan2-ada-pytorch data/processed/mendeley_256 datasets/pneumonia_256_conditional.zip

REPO_DIR="${1:-third_party/stylegan2-ada-pytorch}"
SOURCE_DIR="${2:-data/processed/mendeley_256}"
DEST_ZIP="${3:-datasets/pneumonia_256_conditional.zip}"

mkdir -p "$(dirname "$DEST_ZIP")"

python "$REPO_DIR/dataset_tool.py" \
  --source "$SOURCE_DIR" \
  --dest "$DEST_ZIP" \
  --width 256 \
  --height 256

echo "Created dataset zip at $DEST_ZIP"
