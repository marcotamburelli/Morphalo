#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-$HOME/models/mhr}"
mkdir -p -- "$model_dir"
temp_dir=$(mktemp -d "$model_dir/.fetch.XXXXXXXX")
trap 'rm -rf -- "$temp_dir"' EXIT

rig_sha=352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc
archive_sha=e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94

verify() {
  if ! printf '%s  %s\n' "$1" "$2" | sha256sum --check --status; then
    echo "Checksum verification failed for $2" >&2
    return 1
  fi
}

if [[ -e "$model_dir/mhr_model.pt" ]]; then
  verify "$rig_sha" "$model_dir/mhr_model.pt"
else
  curl --fail --location --retry 3 -o "$temp_dir/assets.zip" \
    https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  verify "$archive_sha" "$temp_dir/assets.zip"
  unzip -p "$temp_dir/assets.zip" assets/mhr_model.pt > "$temp_dir/mhr_model.pt"
  verify "$rig_sha" "$temp_dir/mhr_model.pt"
  mv -- "$temp_dir/mhr_model.pt" "$model_dir/mhr_model.pt"
fi

echo "MHR TorchScript rig ready in $model_dir/mhr_model.pt"
