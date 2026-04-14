#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
MODEL_KIND=${MODEL_KIND:-qwen}
DATASETS=${DATASETS:-"2wikimultihopqa"}
FORCE_REGEN=${FORCE_REGEN:-0}

CKPT_EXPERIMENT_NAME=${CKPT_EXPERIMENT_NAME:-}
STEP=${STEP:-}
CKPT_ROOT=${CKPT_ROOT:-${CKPT_EXPERIMENT_NAME:+$PROJECT_ROOT/verl_checkpoints_3b/$CKPT_EXPERIMENT_NAME}}

EVAL_NAME=${EVAL_NAME:-${CKPT_EXPERIMENT_NAME:-manual-checkpoint}${STEP:+-step$STEP}}
EVAL_ROOT=${EVAL_ROOT:-$PROJECT_ROOT/eval_official/$EVAL_NAME}
DATA_ROOT=${DATA_ROOT:-$EVAL_ROOT/data}
LOG_ROOT=${LOG_ROOT:-$EVAL_ROOT/logs}

TRAIN_FILE=${TRAIN_FILE:-}
TRAIN_SEARCH_ROOT=${TRAIN_SEARCH_ROOT:-$PROJECT_ROOT/data}

NUM_GPUS=${NUM_GPUS:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_START_LENGTH=${MAX_START_LENGTH:-2560}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-900}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
MAX_TURNS=${MAX_TURNS:-5}
REWARD_METRIC=${REWARD_METRIC:-em}

CONFIG_DIR=${CONFIG_DIR:-$HOME/.config/router-r1}

find_latest_train_file() {
  local pattern="train_nh_${MODEL_KIND}.parquet"
  find "$TRAIN_SEARCH_ROOT" -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

resolve_checkpoint_path() {
  local role="$1"
  local explicit_path="$2"
  if [ -n "$explicit_path" ]; then
    printf '%s\n' "$explicit_path"
    return 0
  fi

  if [ -z "${CKPT_ROOT:-}" ]; then
    return 1
  fi

  if [ -n "$STEP" ]; then
    printf '%s\n' "$CKPT_ROOT/$role/global_step_$STEP"
    return 0
  fi

  ls -1dt "$CKPT_ROOT/$role"/global_step_* 2>/dev/null | head -n 1
}

PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
if [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] python not found in PATH. Activate the target conda env first."
  exit 1
fi

[ -f "$CONFIG_DIR/openrouter.env" ] && set -a && source "$CONFIG_DIR/openrouter.env" && set +a
[ -f "$CONFIG_DIR/wandb.env" ] && set -a && source "$CONFIG_DIR/wandb.env" && set +a

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[ERROR] OPENROUTER_API_KEY is not set."
  exit 1
fi

ACTOR_PATH=${ACTOR_PATH:-$(resolve_checkpoint_path actor "${ACTOR_PATH:-}" || true)}
CRITIC_PATH=${CRITIC_PATH:-$(resolve_checkpoint_path critic "${CRITIC_PATH:-}" || true)}

if [ -z "$ACTOR_PATH" ] || [ ! -d "$ACTOR_PATH" ]; then
  echo "[ERROR] Invalid actor checkpoint path: ${ACTOR_PATH:-<empty>}"
  exit 1
fi

if [ -z "$CRITIC_PATH" ] || [ ! -d "$CRITIC_PATH" ]; then
  echo "[ERROR] Invalid critic checkpoint path: ${CRITIC_PATH:-<empty>}"
  exit 1
fi

if [ -z "$TRAIN_FILE" ]; then
  TRAIN_FILE=$(find_latest_train_file)
fi

if [ -z "$TRAIN_FILE" ] || [ ! -f "$TRAIN_FILE" ]; then
  echo "[ERROR] Could not resolve TRAIN_FILE. Set TRAIN_FILE explicitly or place train_nh_${MODEL_KIND}.parquet under $TRAIN_SEARCH_ROOT."
  exit 1
fi

mkdir -p "$DATA_ROOT" "$LOG_ROOT"

echo "[EVAL] PROJECT_ROOT=$PROJECT_ROOT"
echo "[EVAL] EVAL_NAME=$EVAL_NAME"
echo "[EVAL] MODEL_KIND=$MODEL_KIND"
echo "[EVAL] DATASETS=$DATASETS"
echo "[EVAL] TRAIN_FILE=$TRAIN_FILE"
echo "[EVAL] ACTOR_PATH=$ACTOR_PATH"
echo "[EVAL] CRITIC_PATH=$CRITIC_PATH"
echo "[EVAL] EVAL_ROOT=$EVAL_ROOT"
echo "[EVAL] MAX_TURNS=$MAX_TURNS"

pushd "$PROJECT_ROOT" >/dev/null

for dataset in $DATASETS; do
  DATA_DIR="$DATA_ROOT/$dataset"
  TEST_PARQUET="$DATA_DIR/test_${dataset}_${MODEL_KIND}.parquet"
  LOG_FILE="$LOG_ROOT/${dataset}.log"

  if [ "$FORCE_REGEN" = "1" ] && [ -f "$TEST_PARQUET" ]; then
    rm -f "$TEST_PARQUET"
  fi

  if [ -f "$TEST_PARQUET" ]; then
    echo "[DATA] Reusing existing test parquet: $TEST_PARQUET"
  else
    echo "[DATA] Generating test parquet for $dataset with current branch prompt"
    mkdir -p "$DATA_DIR"
    "$PYTHON_BIN" data_process/qa_test_gen.py \
      --data_sources "$dataset" \
      --model "$MODEL_KIND" \
      --local_dir "$DATA_DIR"
  fi

  if [ ! -f "$TEST_PARQUET" ]; then
    echo "[ERROR] Missing generated test parquet: $TEST_PARQUET"
    exit 1
  fi

  echo "[RUN] dataset=$dataset log=$LOG_FILE"
  PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "$PYTHON_BIN" -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_PARQUET" \
    data.train_batch_size=$VAL_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$ACTOR_PATH" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$VAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="$CRITIC_PATH" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    critic.ppo_micro_batch_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.logger='["console"]' \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$EVAL_ROOT/$dataset" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1000000 \
    trainer.test_freq=1000000 \
    trainer.project_name=router-r1-eval \
    trainer.experiment_name="$EVAL_NAME-$dataset" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    max_turns=$MAX_TURNS \
    +reward_metric="$REWARD_METRIC" \
    +cost_coe=0.0 \
    +api_base="'${OPENROUTER_API_BASE:-https://openrouter.ai/api/v1}'" \
    +api_key="'$OPENROUTER_API_KEY'" \
    2>&1 | tee "$LOG_FILE"
done

popd >/dev/null