#!/usr/bin/env bash
# Linux y macOS: instala uv si falta y ejecuta el instalador (install.py).
# Uso: ./setup.sh [--no-models]
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo "Instalando uv (gestor de Python de Astral) en ~/.local/bin..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv run --no-project --python 3.12 install.py "$@"
