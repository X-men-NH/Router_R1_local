#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/autodl-tmp/Router-R1}
ENV_NAME=${ENV_NAME:-router-r1}
PY_VER=${PY_VER:-3.9}
SHARED_ROOT=${SHARED_ROOT:-$(cd -- "$REPO_DIR/.." && pwd)/router-r1-shared}
CONFIG_DIR=${CONFIG_DIR:-"$SHARED_ROOT/config"}

if [ ! -d "$REPO_DIR" ]; then
  echo "[ERROR] REPO_DIR not found: $REPO_DIR"
  echo "[HINT] set REPO_DIR or clone repo first"
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

cd "$REPO_DIR"
pip install -r requirements.txt
pip install -e .

mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/openrouter.env" ]; then
  cat > "$CONFIG_DIR/openrouter.env" <<'EOF'
export OPENROUTER_API_KEY='replace_with_your_real_openrouter_key'
export OPENROUTER_API_BASE='https://openrouter.ai/api/v1'
EOF
  chmod 600 "$CONFIG_DIR/openrouter.env"
  echo "[INFO] Created $CONFIG_DIR/openrouter.env (fill real key)"
fi

if [ ! -f "$CONFIG_DIR/wandb.env" ]; then
  cat > "$CONFIG_DIR/wandb.env" <<'EOF'
export WANDB_API_KEY='replace_with_your_real_wandb_key'
EOF
  chmod 600 "$CONFIG_DIR/wandb.env"
  echo "[INFO] Created $CONFIG_DIR/wandb.env (optional)"
fi

echo "[OK] Environment ready in conda env: $ENV_NAME"
