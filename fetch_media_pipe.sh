#!/usr/bin/env bash
set -e

BASE_DIR="$HOME/models/mediapipe"

mkdir -p "$BASE_DIR"

download_if_missing () {
  local OUT="$1"
  local URL="$2"

  if [ -f "$OUT" ]; then
    echo "$(basename "$OUT") already exists, skipping"
  else
    echo "Downloading $(basename "$OUT")"
    wget -O "$OUT" "$URL"
  fi
}

download_if_missing "$BASE_DIR/pose_landmarker_heavy.task" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

download_if_missing "$BASE_DIR/hand_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

download_if_missing "$BASE_DIR/face_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

echo "MediaPipe models ready in $BASE_DIR"
