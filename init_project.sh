#!/usr/bin/env bash
set -e

# Project root (absolute path)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Project-local virtual environment dir
VENV_DIR="$PROJECT_ROOT/venv"

# Python executable
PYTHON=python3

# Flags
REBUILD=false

# Parse CLI args
for arg in "$@"; do
  case "$arg" in
    --rebuild)
      REBUILD=true
      ;;
    -h|--help)
      echo "Usage: ./init_project.sh [--rebuild]"
      echo "  --rebuild   Remove and recreate ./venv from scratch"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Try: ./init_project.sh --help"
      exit 1
      ;;
  esac
done

echo "== ML_Startup initialization =="

# If --rebuild is set, remove the existing venv first
if [ "$REBUILD" = true ] && [ -d "$VENV_DIR" ]; then
  echo ">> Rebuilding virtual environment (removing existing venv)"
  rm -rf "$VENV_DIR"
fi

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo ">> Creating virtual environment"
  $PYTHON -m venv "$VENV_DIR"
else
  echo ">> Virtual environment already exists"
fi

# Activate venv
echo ">> Activating virtual environment"
source "$VENV_DIR/bin/activate"

# Upgrade packaging tools
echo ">> Upgrading pip, setuptools and wheel"
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.8)
# NOTE: PyTorch wheels include the CUDA runtime.
echo ">> Installing PyTorch (CUDA 12.8)"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install project requirements (if present)
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
  echo ">> Installing project requirements"
  python -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

# Quick sanity check (non-blocking)
echo ">> Verifying CUDA availability"
python - << 'EOF' || true
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "== Bootstrap completed successfully =="
