#!/usr/bin/env bash
set -e

BASE_DIR="$HOME/models/sam"

mkdir -p "$BASE_DIR"

download_if_missing () {
  local OUT="$1"
  local URL="$2"

  if [ -f "$OUT" ]; then
    echo "$OUT already exists, skipping"
  else
    echo "Downloading $(basename "$OUT")"
    wget -O "$OUT" "$URL"
  fi
}

download_if_missing "$BASE_DIR/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

download_if_missing "$BASE_DIR/sam_vit_l_0b3195.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

download_if_missing "$BASE_DIR/sam_vit_b_01ec64.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "SAM models ready in $BASE_DIR"
