#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PUBLIC_ROOT=$(cd -- "$PROJECT_ROOT/.." && pwd)
SHARED_ROOT=${SHARED_ROOT:-"$PUBLIC_ROOT/router-r1-shared"}

MINICONDA_PREFIX=${MINICONDA_PREFIX:-"$PUBLIC_ROOT/miniconda3"}
MINICONDA_INSTALLER=${MINICONDA_INSTALLER:-"$PUBLIC_ROOT/Miniconda3-latest-Linux-x86_64.sh"}
ENV_NAME=${ENV_NAME:-router-r1}
PYTHON_VERSION=${PYTHON_VERSION:-3.9}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
TORCH_VERSION=${TORCH_VERSION:-2.4.0}
VLLM_VERSION=${VLLM_VERSION:-0.6.3}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.6.3}
CONFIG_DIR=${CONFIG_DIR:-"$SHARED_ROOT/config"}

log() {
    printf '[INFO] %s\n' "$1"
}

warn() {
    printf '[WARN] %s\n' "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf '[ERROR] Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

download_miniconda() {
    if [ -x "$MINICONDA_PREFIX/bin/conda" ]; then
        log "Miniconda already present at $MINICONDA_PREFIX"
        return
    fi

    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        log "Downloading Miniconda installer to $MINICONDA_INSTALLER"
        if command -v wget >/dev/null 2>&1; then
            wget -O "$MINICONDA_INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        elif command -v curl >/dev/null 2>&1; then
            curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MINICONDA_INSTALLER"
        else
            printf '[ERROR] Neither wget nor curl is available for downloading Miniconda.\n' >&2
            exit 1
        fi
    fi

    log "Installing Miniconda into $MINICONDA_PREFIX"
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_PREFIX"
}

activate_conda() {
    # shellcheck source=/dev/null
    source "$MINICONDA_PREFIX/etc/profile.d/conda.sh"
}

accept_conda_tos() {
    activate_conda
    if conda tos --help >/dev/null 2>&1; then
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
    fi
}

ensure_env() {
    accept_conda_tos
    activate_conda
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        log "Creating conda environment $ENV_NAME with Python $PYTHON_VERSION"
        conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    else
        log "Conda environment $ENV_NAME already exists"
    fi
    conda activate "$ENV_NAME"
}

install_python_packages() {
    log "Upgrading pip toolchain"
    python -m pip install -U pip setuptools wheel packaging ninja

    log "Installing PyTorch $TORCH_VERSION from $TORCH_INDEX_URL"
    python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

    log "Installing vLLM $VLLM_VERSION"
    python -m pip install "vllm==$VLLM_VERSION"

    local filtered_requirements
    filtered_requirements=$(mktemp)
    python - "$PROJECT_ROOT/requirements.txt" "$filtered_requirements" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
skip_prefixes = ("vllm", "flash-attn")

lines = []
for raw_line in source.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith(skip_prefixes):
        continue
    lines.append(line)

target.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

    log "Installing remaining requirements"
    python -m pip install -r "$filtered_requirements"
    rm -f "$filtered_requirements"

    log "Installing flash-attn $FLASH_ATTN_VERSION"
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1 python -m pip install --no-cache-dir --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"

    log "Installing project in editable mode"
    cd "$PROJECT_ROOT"
    python -m pip install -e . --no-deps
}

ensure_config_templates() {
    mkdir -p "$CONFIG_DIR"

    if [ ! -f "$CONFIG_DIR/openrouter.env" ]; then
        cat > "$CONFIG_DIR/openrouter.env" <<'EOF'
export OPENROUTER_API_KEY='replace_with_your_real_openrouter_key'
export OPENROUTER_API_BASE='https://openrouter.ai/api/v1'
EOF
        chmod 600 "$CONFIG_DIR/openrouter.env"
        warn "Created $CONFIG_DIR/openrouter.env with placeholder values"
    fi

    if [ ! -f "$CONFIG_DIR/wandb.env" ]; then
        cat > "$CONFIG_DIR/wandb.env" <<'EOF'
export WANDB_API_KEY='replace_with_your_real_wandb_key'
EOF
        chmod 600 "$CONFIG_DIR/wandb.env"
        warn "Created $CONFIG_DIR/wandb.env with placeholder values"
    fi
}

verify_install() {
    log "Verifying core Python imports"
    python - <<'PY'
import datasets
import ray
import torch
import transformers
import verl

print('python_ok')
print(f'torch={torch.__version__}')
print(f'transformers={transformers.__version__}')
print(f'ray={ray.__version__}')
print(f'datasets={datasets.__version__}')
PY

    log "Verifying vLLM import"
    python - <<'PY'
import vllm
print(f'vllm={vllm.__version__}')
PY

    log "Verifying flash-attn import"
    python - <<'PY'
import flash_attn
print(f'flash_attn={flash_attn.__version__}')
PY

    if ! ls /dev/nvidia* >/dev/null 2>&1; then
        warn "No NVIDIA device files are visible on this host. Training/inference will not run until GPUs are attached."
    fi
}

main() {
    require_cmd bash
    require_cmd awk
    download_miniconda
    ensure_env
    install_python_packages
    ensure_config_templates
    verify_install

    cat <<EOF
[OK] Router-R1 environment setup finished.
[NEXT] Activate with:
source "$MINICONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
source "$CONFIG_DIR/openrouter.env"
source "$CONFIG_DIR/wandb.env"
cd "$PROJECT_ROOT"
EOF
}

main "$@"