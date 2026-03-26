#!/usr/bin/env bash
set -euo pipefail

SRC_DIR=${SRC_DIR:-$(pwd)}
DATA_DISK_DIR=${DATA_DISK_DIR:-/root/autodl-tmp/Router-R1}
ENV_NAME=${ENV_NAME:-router-r1}

echo "[INFO] SRC_DIR=$SRC_DIR"
echo "[INFO] DATA_DISK_DIR=$DATA_DISK_DIR"

mkdir -p "$DATA_DISK_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -av --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='.venv/' \
    --exclude='venv/' \
    "$SRC_DIR/" "$DATA_DISK_DIR/"
else
  cp -a "$SRC_DIR/." "$DATA_DISK_DIR/"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  fi
fi

python -m pip freeze > "$DATA_DISK_DIR/requirements.lock.txt"

if command -v conda >/dev/null 2>&1; then
  conda env export --name "$ENV_NAME" --no-builds > "$DATA_DISK_DIR/conda-env.$ENV_NAME.yml" || true
fi

cat > "$DATA_DISK_DIR/RESTORE_FROM_DATADISK.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/autodl-tmp/Router-R1}
ENV_NAME=${ENV_NAME:-router-r1}

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
  conda create -y -n "$ENV_NAME" python=3.9
fi

conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

if [ -f "$REPO_DIR/requirements.lock.txt" ]; then
  pip install -r "$REPO_DIR/requirements.lock.txt"
else
  pip install -r "$REPO_DIR/requirements.txt"
fi

cd "$REPO_DIR"
pip install -e .

echo "[OK] Restore complete."
echo "[NEXT] source ~/.config/router-r1/openrouter.env && bash train_3b_isolated.sh"
EOF
chmod +x "$DATA_DISK_DIR/RESTORE_FROM_DATADISK.sh"

echo "[OK] Saved project and dependency locks to data disk."
echo "[OK] requirements.lock.txt: $DATA_DISK_DIR/requirements.lock.txt"
echo "[OK] restore script: $DATA_DISK_DIR/RESTORE_FROM_DATADISK.sh"
