#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
ROCM_TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3"

# ── helpers ──────────────────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
die()    { red "ERROR: $*"; exit 1; }

# ── checks ───────────────────────────────────────────────────────────────────
command -v python3 &>/dev/null || die "python3 not found"
python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" \
    || die "Python 3.10+ required (found $(python3 --version))"

# ── virtual environment ───────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    yellow "Reusing existing virtualenv at $VENV_DIR"
else
    green "Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

green "Upgrading pip"
pip install --quiet --upgrade pip

# ── PyTorch (ROCm build) ──────────────────────────────────────────────────────
if python3 -c "import torch" &>/dev/null; then
    yellow "PyTorch already installed ($(python3 -c 'import torch; print(torch.__version__)'))"
else
    green "Installing PyTorch with ROCm support"
    pip install torch torchvision --index-url "$ROCM_TORCH_INDEX"
fi

# ── project (editable) ───────────────────────────────────────────────────────
green "Installing project in editable mode with dev extras"
pip install -e ".[dev]"

# ── verify ───────────────────────────────────────────────────────────────────
green "Verifying setup"

python3 - <<'EOF'
import torch
from plant_classifier import PlantClassifier, MetaCLIPEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU : {name}  ({vram_gb:.1f} GB VRAM)")
else:
    print("  GPU : not available — running on CPU")

print(f"  PyTorch : {torch.__version__}")
print(f"  Device  : {device}")
print("  Imports : OK")
EOF

green "Done — activate with: source $VENV_DIR/bin/activate"
