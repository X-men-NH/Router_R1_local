export CUDA_VISIBLE_DEVICES=2,3,4,5
export DATA_DIR='data/nq_search'

WAND_PROJECT='Router-R1-Official'

#export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
#export EXPERIMENT_NAME=nh-bs64-ppo-llama3.2-3b-it-em
#export BASE_MODEL='/home/workspace/xnh/models/Qwen2.5-3B-Instruct'
#export EXPERIMENT_NAME=nh-bs64-ppo-qwen2.5-3b-it-em
export BASE_MODEL='/home/workspace/xnh/models/Qwen2.5-1.5B-Instruct'
export EXPERIMENT_NAME=nh-bs32-ppo-qwen2.5-1.5b-it-em

TARGET_TOTAL_STEPS=225
CKPT_ROOT="/home/workspace/xnh/Router-R1/verl_checkpoints/$EXPERIMENT_NAME"

# Auto-resume from checkpoint whose internal files were modified most recently.
# This is more reliable than directory mtime because overwriting existing files
# may not update the directory timestamp.
latest_ckpt_by_file_mtime() {
    local parent_dir="$1"
    if [ ! -d "$parent_dir" ]; then
        return
    fi

    local latest_dir
    latest_dir=$(find "$parent_dir" -maxdepth 2 -type f -path "*/global_step_*/*" -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr \
        | awk '!seen[$2]++ {print $2; exit}')

    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
    else
        ls -1dt "$parent_dir/global_step_"* 2>/dev/null | head -n 1
    fi
}

LATEST_ACTOR_CKPT=$(latest_ckpt_by_file_mtime "$CKPT_ROOT/actor")
LATEST_CRITIC_CKPT=$(latest_ckpt_by_file_mtime "$CKPT_ROOT/critic")

if [ -n "$LATEST_ACTOR_CKPT" ] && [ -n "$LATEST_CRITIC_CKPT" ]; then
    export ACTOR_CKPT="$LATEST_ACTOR_CKPT"
    export CRITIC_CKPT="$LATEST_CRITIC_CKPT"
    RESUME_STEP=$(basename "$ACTOR_CKPT" | sed 's/global_step_//')
    REMAINING_STEPS=$((TARGET_TOTAL_STEPS - RESUME_STEP))
    if [ "$REMAINING_STEPS" -lt 1 ]; then
        REMAINING_STEPS=1
    fi
else
    # Fallback when no checkpoint exists yet.
    export ACTOR_CKPT="$BASE_MODEL"
    export CRITIC_CKPT="$BASE_MODEL"
    RESUME_STEP=0
    REMAINING_STEPS=$TARGET_TOTAL_STEPS
fi

echo "[RESUME] ACTOR_CKPT=$ACTOR_CKPT"
echo "[RESUME] CRITIC_CKPT=$CRITIC_CKPT"
echo "[RESUME] RESUME_STEP=$RESUME_STEP, REMAINING_STEPS=$REMAINING_STEPS"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-1800}
export ROUTER_API_TIMEOUT=${ROUTER_API_TIMEOUT:-30}
export ROUTER_API_MAX_RETRIES=${ROUTER_API_MAX_RETRIES:-2}
export ROUTER_API_MAX_TRIALS=${ROUTER_API_MAX_TRIALS:-6}
export ROUTER_API_RETRY_GAP_SEC=${ROUTER_API_RETRY_GAP_SEC:-1.5}
export ROUTER_LOG_QUESTIONS=${ROUTER_LOG_QUESTIONS:-1}
export ROUTER_LOG_MAX_QUESTIONS=${ROUTER_LOG_MAX_QUESTIONS:-3}
export ROUTER_LOG_MAX_CHARS=${ROUTER_LOG_MAX_CHARS:-240}
export ROUTER_LOG_ROUTE_TRACE=${ROUTER_LOG_ROUTE_TRACE:-1}
export ROUTER_LOG_MAX_EVENTS=${ROUTER_LOG_MAX_EVENTS:-5}
export ROUTER_SKIP_STALLED_STEP=${ROUTER_SKIP_STALLED_STEP:-1}
export ROUTER_MAX_STALLED_TURNS=${ROUTER_MAX_STALLED_TURNS:-2}

# Reduce glibc memory fragmentation — makes malloc return memory to OS faster.
# This helps prevent system OOM when FSDP offload puts large tensors on CPU RAM.
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072

#data.train_files=$DATA_DIR/train_nh_llama.parquet \
#data.val_files=$DATA_DIR/test_nh_llama.parquet \

# Attention: DataLoader is set to drop_last=True by default, please set data.val_batch_size to a reasonable value.

RAY_memory_usage_threshold=0.99 PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/workspace/xnh/anaconda3/envs/router-r1/bin/python -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_nh_qwen.parquet \
    data.val_files=$DATA_DIR/test_nh_qwen.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=3072 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.path=$CRITIC_CKPT \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=15 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$REMAINING_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    +reward_metric="em" \
    +cost_coe=0.0 \
    +api_base="https://openrouter.ai/api/v1" \
    +api_key="[YOU_API_KEY]" \
    2>&1 | tee $EXPERIMENT_NAME.log
