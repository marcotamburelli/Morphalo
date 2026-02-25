#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dag-module> [-- <args passed to the dag script>]" >&2
  exit 1
fi

DAG_MODULE="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

cd "$PROJECT_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "X Virtual environment not found at:"
  echo "   $VENV_DIR" >&2
  echo "Run ./init_project.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# python -m "$DAG_MODULE" "$@"
python -m stability.cli run-dags "$DAG_MODULE" "$@"
