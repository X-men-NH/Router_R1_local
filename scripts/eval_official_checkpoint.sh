#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"$PROJECT_ROOT/data/nq_search_main_clean_0414"}
CKPT_EXPERIMENT_NAME=${CKPT_EXPERIMENT_NAME:-nh-bs64-ppo-qwen2.5-3b-it-em-main-clean-0414-1gpu-01}
STEP=${STEP:-210}
MODEL_KIND=${MODEL_KIND:-qwen}
DATASETS=${DATASETS:-"2wikimultihopqa musique bamboogle"}
EVAL_ROOT=${EVAL_ROOT:-"$PROJECT_ROOT/eval_official_main/${CKPT_EXPERIMENT_NAME}-step${STEP}"}
FORCE_REGEN=${FORCE_REGEN:-0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

ACTOR_PATH=${ACTOR_PATH:-"$PROJECT_ROOT/verl_checkpoints_3b/$CKPT_EXPERIMENT_NAME/actor/global_step_${STEP}"}
CRITIC_PATH=${CRITIC_PATH:-"$PROJECT_ROOT/verl_checkpoints_3b/$CKPT_EXPERIMENT_NAME/critic/global_step_${STEP}"}

VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-4}
ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-1}
CRITIC_PPO_MICRO_BATCH_SIZE=${CRITIC_PPO_MICRO_BATCH_SIZE:-1}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-1}
REF_LOGPROB_MICRO_BATCH_SIZE=${REF_LOGPROB_MICRO_BATCH_SIZE:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}
MAX_TURNS=${MAX_TURNS:-4}

PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python || true)}
if [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] python3/python not found in PATH. Activate the router-r1 environment first."
  exit 1
fi

[ -f "$HOME/.config/router-r1/openrouter.env" ] && source "$HOME/.config/router-r1/openrouter.env"
[ -f "$HOME/.config/router-r1/wandb.env" ] && source "$HOME/.config/router-r1/wandb.env"

trim_trailing_cr() {
  local value="${1-}"
  printf '%s' "${value%$'\r'}"
}

OPENROUTER_API_KEY=$(trim_trailing_cr "${OPENROUTER_API_KEY-}")
OPENROUTER_API_BASE=$(trim_trailing_cr "${OPENROUTER_API_BASE-}")
export OPENROUTER_API_KEY OPENROUTER_API_BASE

if [ ! -f "$TRAIN_DATA_DIR/train_nh_qwen.parquet" ]; then
  echo "[ERROR] Missing train file: $TRAIN_DATA_DIR/train_nh_qwen.parquet"
  exit 1
fi

if [ ! -d "$ACTOR_PATH" ]; then
  echo "[ERROR] Missing actor checkpoint: $ACTOR_PATH"
  exit 1
fi

if [ ! -d "$CRITIC_PATH" ]; then
  echo "[ERROR] Missing critic checkpoint: $CRITIC_PATH"
  exit 1
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "[ERROR] OPENROUTER_API_KEY is not set."
  exit 1
fi

export OPENROUTER_API_BASE=${OPENROUTER_API_BASE:-"https://openrouter.ai/api/v1"}

mkdir -p "$EVAL_ROOT/data" "$EVAL_ROOT/logs" "$PROJECT_ROOT/verl_checkpoints_eval"

echo "[OFFICIAL-EVAL] PROJECT_ROOT=$PROJECT_ROOT"
echo "[OFFICIAL-EVAL] TRAIN_DATA_DIR=$TRAIN_DATA_DIR"
echo "[OFFICIAL-EVAL] CKPT_EXPERIMENT_NAME=$CKPT_EXPERIMENT_NAME"
echo "[OFFICIAL-EVAL] STEP=$STEP"
echo "[OFFICIAL-EVAL] ACTOR_PATH=$ACTOR_PATH"
echo "[OFFICIAL-EVAL] CRITIC_PATH=$CRITIC_PATH"
echo "[OFFICIAL-EVAL] DATASETS=$DATASETS"
echo "[OFFICIAL-EVAL] EVAL_ROOT=$EVAL_ROOT"

pushd "$PROJECT_ROOT" >/dev/null

for dataset in $DATASETS; do
  DATA_DIR="$EVAL_ROOT/data/$dataset"
  VAL_FILE="$DATA_DIR/test_${dataset}_${MODEL_KIND}.parquet"
  RUN_NAME="official-val-${dataset}-step${STEP}"
  LOG_FILE="$EVAL_ROOT/logs/${RUN_NAME}.log"

  mkdir -p "$DATA_DIR"

  if [ "$FORCE_REGEN" = "1" ] || [ ! -f "$VAL_FILE" ]; then
    rm -f "$VAL_FILE"
    echo "[OFFICIAL-EVAL] Generating test parquet for $dataset"
    "$PYTHON_BIN" data_process/qa_test_gen.py --data_sources "$dataset" --model "$MODEL_KIND" --local_dir "$DATA_DIR"
  else
    echo "[OFFICIAL-EVAL] Reusing existing parquet for $dataset: $VAL_FILE"
  fi

  if [ ! -f "$VAL_FILE" ]; then
    echo "[ERROR] Missing validation parquet: $VAL_FILE"
    exit 1
  fi

  echo "[OFFICIAL-EVAL] Starting val-only run for $dataset"
  PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "$PYTHON_BIN" -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_DATA_DIR/train_nh_qwen.parquet" \
    data.val_files="$VAL_FILE" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$ACTOR_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ACTOR_PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$ACTOR_PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.path="$CRITIC_PATH" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size="$CRITIC_PPO_MICRO_BATCH_SIZE" \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.logger="['console']" \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.project_name=Router-R1-Official-Recheck \
    trainer.experiment_name="$RUN_NAME" \
    trainer.total_epochs=100 \
    trainer.total_training_steps=225 \
    trainer.default_local_dir="$PROJECT_ROOT/verl_checkpoints_eval/$RUN_NAME" \
    max_turns="$MAX_TURNS" \
    +reward_metric="em" \
    +cost_coe=0.0 \
    +api_base="$OPENROUTER_API_BASE" \
    +api_key="$OPENROUTER_API_KEY" \
    2>&1 | tee "$LOG_FILE"
done

popd >/dev/null