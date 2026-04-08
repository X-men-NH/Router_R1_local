#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-nh-bs64-ppo-qwen2.5-3b-it-em-neutralprompt-0406-01}
MODEL_KIND=${MODEL_KIND:-qwen}
DATASETS=${DATASETS:-"2wikimultihopqa musique bamboogle"}
EVAL_ROOT=${EVAL_ROOT:-"$PROJECT_ROOT/eval_multihop/$EXPERIMENT_NAME"}
DATA_ROOT=${DATA_ROOT:-"$EVAL_ROOT/data"}
CKPT_ROOT=${CKPT_ROOT:-"$PROJECT_ROOT/verl_checkpoints_3b/$EXPERIMENT_NAME"}
MODEL_PATH=${MODEL_PATH:-$(ls -1dt "$CKPT_ROOT/actor/global_step_"* 2>/dev/null | head -n 1)}

PROMPT_LENGTH=${PROMPT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
TEMPERATURE=${TEMPERATURE:-1.0}
N_SAMPLES=${N_SAMPLES:-1}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-1}

if [ -z "$MODEL_PATH" ]; then
  echo "[ERROR] Could not find actor checkpoint under $CKPT_ROOT/actor"
  exit 1
fi

echo "[EVAL] PROJECT_ROOT=$PROJECT_ROOT"
echo "[EVAL] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[EVAL] MODEL_PATH=$MODEL_PATH"
echo "[EVAL] DATASETS=$DATASETS"
echo "[EVAL] EVAL_ROOT=$EVAL_ROOT"
echo "[EVAL] DATA_ROOT=$DATA_ROOT"

mkdir -p "$DATA_ROOT" "$EVAL_ROOT/generated"

pushd "$PROJECT_ROOT" >/dev/null

for dataset in $DATASETS; do
  DATA_DIR="$DATA_ROOT/$dataset"
  GENERATED_PATH="$EVAL_ROOT/generated/${dataset}.parquet"
  TEST_PARQUET="$DATA_DIR/test_${dataset}_${MODEL_KIND}.parquet"

  if [ -f "$TEST_PARQUET" ]; then
    echo "\n==================== $dataset: reuse existing test set ===================="
    echo "[EVAL] Using existing parquet: $TEST_PARQUET"
  else
    echo "\n==================== $dataset: prepare test set ===================="
    mkdir -p "$DATA_DIR"
    python data_process/qa_test_gen.py --data_sources "$dataset" --model "$MODEL_KIND" --local_dir "$DATA_DIR"

    if [ ! -f "$TEST_PARQUET" ]; then
      echo "[ERROR] Missing generated test parquet: $TEST_PARQUET"
      exit 1
    fi
  fi

  echo "==================== $dataset: generate responses ===================="
  PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m verl.trainer.main_generation \
    data.path="$TEST_PARQUET" \
    data.prompt_key=prompt \
    data.n_samples="$N_SAMPLES" \
    data.output_path="$GENERATED_PATH" \
    data.batch_size="$GEN_BATCH_SIZE" \
    model.path="$MODEL_PATH" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    ++actor.ulysses_sequence_parallel_size=1 \
    rollout.name=vllm \
    ++rollout.n=1 \
    rollout.temperature="$TEMPERATURE" \
    rollout.prompt_length="$PROMPT_LENGTH" \
    rollout.response_length="$RESPONSE_LENGTH" \
    rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    rollout.tensor_model_parallel_size="$TP_SIZE"

  echo "==================== $dataset: evaluate ===================="
  python scripts/evaluate_qa_generation.py --input_parquet "$GENERATED_PATH"
done

popd >/dev/null