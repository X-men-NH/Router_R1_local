# Granule-R1 

`Granule-R1` is a local research project for decomposition-aware multi-hop LLM routing. It trains and evaluates a 3B router that can break a question into subquestions, route each step to external LLMs through OpenRouter, aggregate the intermediate evidence, and produce a final answer.

This README describes this local project only. It does not use the upstream `train.sh` / `test.sh` workflow.

## What This Project Adds

- **Decomposition-aware actions**: the rollout loop supports `<decompose>`, `<subanswer>`, `<search>`, and `<answer>` actions.
- **Action-only scoring**: training and evaluation can score the action trace through `action_responses` instead of unrelated generated text.
- **Hybrid reward**: the default training reward uses EM as the main metric and adds a capped F1 shaping bonus for partially correct answers.
- **Local 3B training workflow**: the main launcher is `train_3b_isolated_a10080_fast.sh`.
- **Checkpoint evaluation and trace analysis**: scripts are provided for validation-style evaluation, generation-based multihop evaluation, and decomposition trace inspection.

## Repository Map

- `train_3b_isolated_a10080_fast.sh`: main 3B PPO training launcher.
- `router_r1/llm_agent/generation.py`: multi-turn routing loop and decomposition action handling.
- `verl/utils/reward_score/qa_em.py`: EM/F1/hybrid answer reward logic.
- `verl/utils/action_trajectory.py`: helpers for extracting action responses.
- `verl/trainer/main_ppo.py`: PPO entrypoint with local reward wiring.
- `verl/trainer/main_generation.py`: generation entrypoint that writes `responses` and `action_responses`.
- `scripts/eval_official_checkpoint.sh`: validation-style checkpoint eval.
- `scripts/eval_multihop_checkpoint.sh`: generation plus QA scoring on multihop datasets.
- `scripts/evaluate_qa_generation.py`: aggregate EM/F1, decomposition rate, search count, and model usage from generated parquet files.
- `scripts/sample_decompose_traces.py`: print sampled decomposition traces.
- `docs/reward_design.md`: detailed notes on the current reward design.
- `NEW_INSTANCE_QUICKSTART.md`: AutoDL restore and shared-config notes.

## Environment Setup

Create and activate the Python environment:

```bash
conda create -n router-r1 python=3.9
conda activate router-r1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

The project expects a CUDA environment that can run vLLM and veRL PPO training. The training scripts also set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` for more predictable single-node behavior.

## Secrets And Config

Do not put API keys in tracked scripts. The training launcher reads environment files from these locations when they exist:

```bash
~/.config/router-r1/openrouter.env
~/.config/router-r1/wandb.env
$CONFIG_DIR/openrouter.env
$CONFIG_DIR/wandb.env
$SHARED_WANDB_FILE
$SHARED_ENV_FILE
$PROJECT_ROOT/.env.local
```

Minimum OpenRouter config:

```bash
mkdir -p ~/.config/router-r1
cat > ~/.config/router-r1/openrouter.env <<'EOF'
export OPENROUTER_API_KEY='replace_with_your_real_openrouter_key'
export OPENROUTER_API_BASE='https://openrouter.ai/api/v1'
EOF
chmod 600 ~/.config/router-r1/openrouter.env
```

Optional W&B config:

```bash
cat > ~/.config/router-r1/wandb.env <<'EOF'
export WANDB_API_KEY='replace_with_your_real_wandb_key'
EOF
chmod 600 ~/.config/router-r1/wandb.env
```

Use `DISABLE_WANDB=1` to force console logging only.

## Data

Training expects Natural Questions / Hotpot-style parquet files under `DATA_DIR`:

```text
$DATA_DIR/train_nh_qwen.parquet
$DATA_DIR/test_nh_qwen.parquet
```

The standard data generation scripts are still available:

```bash
python data_process/qa_train_merge.py --data_sources nq,hotpotqa --model qwen
python data_process/qa_test_merge.py --data_sources nq,hotpotqa --model qwen
python data_process/qa_test_gen.py --data_sources 2wikimultihopqa --model qwen
```

Evaluation scripts can also generate per-dataset test parquet files under their own `eval_*` output directories.

## Train

From the `Granule-R1` directory:

```bash
PROJECT_ROOT="$PWD" \
MODEL_ROOT="/path/to/models" \
DATA_DIR="/path/to/data/nq_search" \
CUDA_VISIBLE_DEVICES=0 \
bash train_3b_isolated_a10080_fast.sh
```

Important environment overrides:

- `PROJECT_ROOT`: repository root for this local project.
- `MODEL_ROOT`: directory containing `Qwen2.5-3B-Instruct`.
- `DATA_DIR`: directory containing `train_nh_qwen.parquet` and `test_nh_qwen.parquet`.
- `FRESH_START=1`: start from the base model with a timestamped run name.
- `EXPERIMENT_NAME=...`: set a stable run name manually.
- `ARTIFACT_ROOT=...`: move logs/checkpoints from the default artifact root.
- `CKPT_ROOT=...`: force a specific checkpoint directory.
- `DISABLE_WANDB=1`: disable W&B.
- `REWARD_METRIC=hybrid|em|f1`: choose answer reward behavior.
- `MAX_TURNS`, `ROUTER_POOL_SIZE`, `ROUTER_API_TIMEOUT`, `SAVE_FREQ`, `TEST_FREQ`: tune rollout and training behavior.

Default outputs:

```text
$PROJECT_ROOT/training_runs/3b/logs/$EXPERIMENT_NAME.log
$PROJECT_ROOT/training_runs/3b/launch_logs/
$PROJECT_ROOT/training_runs/3b/checkpoints/$EXPERIMENT_NAME/
```

The training launcher auto-resumes from the latest actor and critic checkpoints inside `CKPT_ROOT` unless `FRESH_START=1` is set.

## Evaluate

### Validation-Style Checkpoint Eval

Use this when you want to run the trained actor/critic through the PPO validation path:

```bash
CKPT_EXPERIMENT_NAME="your-experiment-name" \
STEP=225 \
DATASETS="2wikimultihopqa" \
bash scripts/eval_official_checkpoint.sh
```

If the checkpoint was produced by `train_3b_isolated_a10080_fast.sh`, point the eval script at the `training_runs` checkpoint layout explicitly:

```bash
CKPT_ROOT="$PWD/training_runs/3b/checkpoints/your-experiment-name" \
TRAIN_FILE="/path/to/train_nh_qwen.parquet" \
bash scripts/eval_official_checkpoint.sh
```

### Generation-Based Multihop Eval

Use this when you want generated parquet outputs and post-hoc QA metrics:

```bash
EXPERIMENT_NAME="your-experiment-name" \
DATASETS="2wikimultihopqa musique bamboogle" \
bash scripts/eval_multihop_checkpoint.sh
```

Generated outputs are written to:

```text
eval_multihop/$EXPERIMENT_NAME/generated/
```

If your actor checkpoint is not under the default `verl_checkpoints_3b` layout, set `MODEL_PATH`:

```bash
MODEL_PATH="$PWD/training_runs/3b/checkpoints/your-experiment-name/actor/global_step_225" \
bash scripts/eval_multihop_checkpoint.sh
```

## Analyze Generated Outputs

Summarize EM/F1, decomposition behavior, search counts, and model usage:

```bash
python scripts/evaluate_qa_generation.py \
  --input_parquet eval_multihop/$EXPERIMENT_NAME/generated/2wikimultihopqa.parquet
```

Sample decomposition traces:

```bash
python scripts/sample_decompose_traces.py \
  --input_parquet eval_multihop/$EXPERIMENT_NAME/generated/2wikimultihopqa.parquet \
  --limit 5
```

Use `--mode random --seed 0` to inspect random examples, or `--full` to print full action traces.

## Reward Summary

The current local reward is documented in `docs/reward_design.md`. At a high level:

```text
reward_total = reward_base + decompose_aux_reward
```

With the default `REWARD_METRIC=hybrid`:

- EM remains the primary metric.
- F1 can add a small shaping bonus when the final answer is partially correct.
- Format penalties are capped for fully correct final answers.
- API cost is computed and logged, but only affects reward when `cost_coe > 0`.
- Decomposition behavior can receive an auxiliary shaping term.

## Routing Pool

Candidate model descriptors live in `data_process/prompt_pool.py`.

When changing the routing pool:

1. Update model descriptors in `data_process/prompt_pool.py`.
2. Update model-name parsing in `router_r1/llm_agent/route_service.py`.
3. Update API pricing in `router_r1/llm_agent/route_service.py`.
4. Regenerate training/eval data if prompt format or candidate model metadata changes.

## Operational Notes

- The default paths in `train_3b_isolated_a10080_fast.sh` are cluster-oriented. Override `PROJECT_ROOT`, `MODEL_ROOT`, and `DATA_DIR` for a local machine.
- Evaluation scripts still have some defaults for the older `verl_checkpoints_3b/$EXPERIMENT_NAME` layout. Prefer explicit `CKPT_ROOT`, `ACTOR_PATH`, `CRITIC_PATH`, or `MODEL_PATH` when evaluating.
- Keep secrets in environment files or runtime environment variables, not in committed scripts.
- Use `NEW_INSTANCE_QUICKSTART.md` for AutoDL restore and shared data/config setup.
