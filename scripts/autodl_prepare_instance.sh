#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/autodl-tmp/Router-R1}
ENV_NAME=${ENV_NAME:-router-r1}
PY_VER=${PY_VER:-3.9}

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

mkdir -p ~/.config/router-r1
if [ ! -f ~/.config/router-r1/openrouter.env ]; then
  cat > ~/.config/router-r1/openrouter.env <<'EOF'
export OPENROUTER_API_KEY='replace_with_your_real_openrouter_key'
export OPENROUTER_API_BASE='https://openrouter.ai/api/v1'
EOF
  chmod 600 ~/.config/router-r1/openrouter.env
  echo "[INFO] Created ~/.config/router-r1/openrouter.env (fill real key)"
fi

if [ ! -f ~/.config/router-r1/wandb.env ]; then
  cat > ~/.config/router-r1/wandb.env <<'EOF'
export WANDB_API_KEY='replace_with_your_real_wandb_key'
EOF
  chmod 600 ~/.config/router-r1/wandb.env
  echo "[INFO] Created ~/.config/router-r1/wandb.env (optional)"
fi

echo "[OK] Environment ready in conda env: $ENV_NAME"
