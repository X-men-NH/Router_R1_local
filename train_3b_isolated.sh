export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT=${PROJECT_ROOT:-'/cpfs01/projects-HDD/cfff-fafaca6b7e53_HDD/public/Router-R1'}
MODEL_ROOT=${MODEL_ROOT:-'/cpfs01/projects-HDD/cfff-fafaca6b7e53_HDD/public/model'}
export DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/data/nq_search"}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

WAND_PROJECT=${WAND_PROJECT:-'Router-R1-3B-Isolated'}

# 3B base model in the migrated project space.
export BASE_MODEL=${BASE_MODEL:-"$MODEL_ROOT/Qwen2.5-3B-Instruct"}

# Use server tag to isolate run/checkpoint/logs across machines.
# You can override manually: export SERVER_TAG=srvB
SERVER_TAG=${SERVER_TAG:-$(hostname -s)}
export EXPERIMENT_NAME=nh-bs64-ppo-qwen2.5-3b-it-em-${SERVER_TAG}

TARGET_TOTAL_STEPS=225
CKPT_ROOT=${CKPT_ROOT:-"$PROJECT_ROOT/verl_checkpoints_3b/$EXPERIMENT_NAME"}

# Auto-resume from latest checkpoint for THIS isolated experiment only.
LATEST_ACTOR_CKPT=$(ls -1dt "$CKPT_ROOT/actor/global_step_"* 2>/dev/null | head -n 1)
LATEST_CRITIC_CKPT=$(ls -1dt "$CKPT_ROOT/critic/global_step_"* 2>/dev/null | head -n 1)

if [ -n "$LATEST_ACTOR_CKPT" ] && [ -n "$LATEST_CRITIC_CKPT" ]; then
    export ACTOR_CKPT="$LATEST_ACTOR_CKPT"
    export CRITIC_CKPT="$LATEST_CRITIC_CKPT"
    RESUME_STEP=$(basename "$ACTOR_CKPT" | sed 's/global_step_//')
    REMAINING_STEPS=$((TARGET_TOTAL_STEPS - RESUME_STEP))
    if [ "$REMAINING_STEPS" -lt 1 ]; then
        REMAINING_STEPS=1
    fi
else
    export ACTOR_CKPT="$BASE_MODEL"
    export CRITIC_CKPT="$BASE_MODEL"
    RESUME_STEP=0
    REMAINING_STEPS=$TARGET_TOTAL_STEPS
fi

echo "[RUN] SERVER_TAG=$SERVER_TAG"
echo "[RUN] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[RESUME] ACTOR_CKPT=$ACTOR_CKPT"
echo "[RESUME] CRITIC_CKPT=$CRITIC_CKPT"
echo "[RESUME] RESUME_STEP=$RESUME_STEP, REMAINING_STEPS=$REMAINING_STEPS"

# Optional: more robust wandb init in unstable networks
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export WANDB_INIT_RETRIES=${WANDB_INIT_RETRIES:-10}
export WANDB_RETRY_SLEEP=${WANDB_RETRY_SLEEP:-20}

PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] python not found in PATH. Please activate your conda env first."
    exit 1
fi

[ -f "$HOME/.config/router-r1/openrouter.env" ] && source "$HOME/.config/router-r1/openrouter.env"
[ -f "$HOME/.config/router-r1/wandb.env" ] && source "$HOME/.config/router-r1/wandb.env"
[ -f "$PROJECT_ROOT/.env.local" ] && set -a && source "$PROJECT_ROOT/.env.local" && set +a

trim_trailing_cr() {
    local value="${1-}"
    printf '%s' "${value%$'\r'}"
}

OPENROUTER_API_KEY=$(trim_trailing_cr "${OPENROUTER_API_KEY-}")
OPENROUTER_API_BASE=$(trim_trailing_cr "${OPENROUTER_API_BASE-}")
WANDB_API_KEY=$(trim_trailing_cr "${WANDB_API_KEY-}")
export OPENROUTER_API_KEY OPENROUTER_API_BASE WANDB_API_KEY

if [ ! -f "$DATA_DIR/train_nh_qwen.parquet" ]; then
    echo "[ERROR] Missing training data: $DATA_DIR/train_nh_qwen.parquet"
    exit 1
fi

if [ ! -f "$DATA_DIR/test_nh_qwen.parquet" ]; then
    echo "[ERROR] Missing validation data: $DATA_DIR/test_nh_qwen.parquet"
    exit 1
fi

export OPENROUTER_API_BASE=${OPENROUTER_API_BASE:-"https://openrouter.ai/api/v1"}
if [ -z "$OPENROUTER_API_KEY" ] || [[ "$OPENROUTER_API_KEY" == *"你的真实完整key"* ]] || [[ "$OPENROUTER_API_KEY" == *"replace_with_your_real_openrouter_key"* ]]; then
    echo "[ERROR] OPENROUTER_API_KEY is not set or still a placeholder."
    echo "[HINT] source ~/.config/router-r1/openrouter.env  (or export OPENROUTER_API_KEY=...)"
    exit 1
fi

TRAINER_LOGGERS=${TRAINER_LOGGERS:-"['wandb']"}
if [ -z "$WANDB_API_KEY" ] || [[ "$WANDB_API_KEY" == *"replace_with_your_real_wandb_key"* ]]; then
    echo "[WARN] WANDB_API_KEY is missing or still a placeholder. Falling back to console logging only."
    TRAINER_LOGGERS="['console']"
fi

RAY_memory_usage_threshold=0.99 PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "$PYTHON_BIN" -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_nh_qwen.parquet \
    data.val_files=$DATA_DIR/test_nh_qwen.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$ACTOR_CKPT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.path=$CRITIC_CKPT \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.logger=$TRAINER_LOGGERS \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$REMAINING_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_ROOT \
    max_turns=4 \
    +reward_metric="em" \
    +cost_coe=0.0 \
    +api_base="'$OPENROUTER_API_BASE'" \
    +api_key="'$OPENROUTER_API_KEY'" \
    2>&1 | tee ${EXPERIMENT_NAME}.log
