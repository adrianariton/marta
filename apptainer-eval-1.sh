#!/bin/bash
#SBATCH --job-name=vllm-bench-a
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-bench-a-%j.log

# ── Models hosted in this job ──────────────────────────────────────────────────
#  Port 8001 │ Qwen2.5-32B-Instruct    │ tensor-parallel=2 (GPUs 0+1, 45% util) ~65GB
#  Port 8002 │ Yi-34B-Chat             │ tensor-parallel=2 (GPUs 0+1, 45% util) ~68GB
#  Port 8003 │ Llama-3.3-70B-Instruct  │ tensor-parallel=3 (all GPUs,  90% util) ~140GB
#
# GPUs 0+1 are shared between Qwen-32B and Yi-34B at 45% each (~72GB each on H100 80G).
# Llama-70B uses all 3 GPUs — vLLM handles multi-process CUDA sharing on H100s fine.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR=$HOME/licenta/models
mkdir -p "$MODEL_DIR"
SIF=$HOME/licenta/vllm-ngc.sif

declare -A MODELS=(
    ["Qwen2.5-32B-Instruct"]="Qwen/Qwen2.5-32B-Instruct"
    ["Yi-34B-Chat"]="01-ai/Yi-34B-Chat"
    ["Llama-3.3-70B-Instruct"]="meta-llama/Llama-3.3-70B-Instruct"
)

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    MODEL_ID="${MODELS[$MODEL_NAME]}"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Downloading $MODEL_ID ..."
        apptainer exec --nv \
            --bind "$MODEL_DIR":/models \
            "$SIF" \
            huggingface-cli download "$MODEL_ID" \
                --local-dir "/models/$MODEL_NAME"
    else
        echo "$MODEL_NAME already present, skipping download."
    fi
done

wait_for_port() {
    local PORT=$1
    echo "Waiting for server on port $PORT ..."
    until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 3
    done
    echo "  → Port $PORT is up!"
}

# ── 1. Qwen2.5-32B — GPUs 0+1, 45% util ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/Qwen2.5-32B-Instruct \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.45 \
        --port 8001 &
PIDS=($!)

wait_for_port 8001

# ── 2. Yi-34B-Chat — GPUs 0+1, 45% util ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/Yi-34B-Chat \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.45 \
        --port 8002 &
PIDS+=($!)

wait_for_port 8002

# ── 3. Llama-3.3-70B — all 3 GPUs, 90% util ──────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 3 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --port 8003 &
PIDS+=($!)

wait_for_port 8003

# ── SSH tunnels ────────────────────────────────────────────────────────────────
for PORT in 8001 8002 8003; do
    ssh -N -o StrictHostKeyChecking=no -R "$PORT:localhost:$PORT" fep &
done

echo ""
echo "=== Job A ready on $(hostname) ==="
echo "  8001 → Qwen2.5-32B-Instruct"
echo "  8002 → Yi-34B-Chat"
echo "  8003 → Llama-3.3-70B-Instruct"

wait "${PIDS[@]}"