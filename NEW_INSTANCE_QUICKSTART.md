# AutoDL New Instance Quickstart (3B)

## 1) Keep everything on data disk

Use this path on AutoDL:

```bash
/root/autodl-tmp/Router-R1
```

On your current instance, run once to save code + dependency locks:

```bash
cd /root/autodl-tmp/Router-R1
bash scripts/autodl_persist_to_datadisk.sh
```

This writes:
- `requirements.lock.txt`
- `conda-env.router-r1.yml` (if conda exists)
- `RESTORE_FROM_DATADISK.sh`

## 2) On a new instance, restore quickly

```bash
cd /root/autodl-tmp/Router-R1
bash RESTORE_FROM_DATADISK.sh
```

## 3) Configure keys (private files)

```bash
mkdir -p /root/autodl-tmp/router-r1-shared/config
cat > /root/autodl-tmp/router-r1-shared/config/openrouter.env <<'EOF'
export OPENROUTER_API_KEY='replace_with_your_real_openrouter_key'
export OPENROUTER_API_BASE='https://openrouter.ai/api/v1'
EOF
chmod 600 /root/autodl-tmp/router-r1-shared/config/openrouter.env
source /root/autodl-tmp/router-r1-shared/config/openrouter.env
```

Optional wandb:

```bash
cat > /root/autodl-tmp/router-r1-shared/config/wandb.env <<'EOF'
export WANDB_API_KEY='replace_with_your_real_wandb_key'
EOF
chmod 600 /root/autodl-tmp/router-r1-shared/config/wandb.env
source /root/autodl-tmp/router-r1-shared/config/wandb.env
```

Shared data directory for multiple branches:

```bash
mkdir -p /root/autodl-tmp/router-r1-shared/data/nq_search
```

## 4) Start 3B training

```bash
cd /root/autodl-tmp/Router-R1
bash train_3b_isolated.sh
```

## Notes

- This quickstart only targets `train_3b_isolated.sh`.
- Keep datasets/checkpoints under `/root/autodl-tmp`.
- Shared keys now live under `/root/autodl-tmp/router-r1-shared/config` so multiple branches can reuse them.
- Shared datasets can live under `/root/autodl-tmp/router-r1-shared/data` so multiple branches can reuse them.
- Do not store real API keys inside repo scripts.
