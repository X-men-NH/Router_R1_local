#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/autodl-tmp/Router-R1}
ENV_NAME=${ENV_NAME:-router-r1}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_3b_isolated.sh}
SHARED_ROOT=${SHARED_ROOT:-$(cd -- "$REPO_DIR/.." && pwd)/router-r1-shared}
CONFIG_DIR=${CONFIG_DIR:-"$SHARED_ROOT/config"}

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

conda activate "$ENV_NAME"

[ -f "$CONFIG_DIR/openrouter.env" ] && source "$CONFIG_DIR/openrouter.env"
[ -f "$CONFIG_DIR/wandb.env" ] && source "$CONFIG_DIR/wandb.env"
[ -f ~/.config/router-r1/openrouter.env ] && source ~/.config/router-r1/openrouter.env
[ -f ~/.config/router-r1/wandb.env ] && source ~/.config/router-r1/wandb.env

if [ -z "${OPENROUTER_API_KEY:-}" ] || [[ "${OPENROUTER_API_KEY:-}" == *"replace_with"* ]]; then
  echo "[ERROR] OPENROUTER_API_KEY missing. Edit $CONFIG_DIR/openrouter.env"
  exit 1
fi

cd "$REPO_DIR"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[ERROR] train script not found: $REPO_DIR/$TRAIN_SCRIPT"
  exit 1
fi

bash "$TRAIN_SCRIPT"
