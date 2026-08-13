#!/bin/bash
#SBATCH --job-name=vllm-bench-b
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-bench-b-%j.log

# ── Models hosted in this job ──────────────────────────────────────────────────
#  Port 8004 │ Mixtral-8x22B-Instruct-v0.1  │ tensor-parallel=3 (all GPUs, 90%) ~140GB
#  Port 8005 │ deepseek-llm-67b-chat         │ tensor-parallel=2 (GPUs 0+1, 45%) ~134GB
#  Port 8006 │ Qwen2.5-72B-Instruct          │ tensor-parallel=3 (all GPUs, 90%) ~145GB
#
# NOTE: Ports 8005 and 8006 / 8004 share GPUs — start sequentially and ensure
# the previous server is fully loaded before launching the next so vLLM can
# correctly negotiate CUDA memory. 70B-class models each need ~2 GPUs minimum.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR=$HOME/licenta/models
mkdir -p "$MODEL_DIR"
SIF=$HOME/licenta/vllm-ngc.sif

declare -A MODELS=(
    ["Mixtral-8x22B-Instruct-v0.1"]="mistralai/Mixtral-8x22B-Instruct-v0.1"
    ["deepseek-llm-67b-chat"]="deepseek-ai/deepseek-llm-67b-chat"
    ["Qwen2.5-72B-Instruct"]="Qwen/Qwen2.5-72B-Instruct"
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

# ── 1. Mixtral-8x22B — GPUs 0+1, 45% util ────────────────────────────────────
# Active params ~39B but full weight load ~141GB; 45% of 2×80GB = 72GB — tight.
# If OOM, drop max-model-len to 16384 or bump to 3 GPUs.
CUDA_VISIBLE_DEVICES=0,1 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/Mixtral-8x22B-Instruct-v0.1 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.45 \
        --port 8004 &
PIDS=($!)

wait_for_port 8004

# ── 2. DeepSeek-67B — GPU 2 + overflow to 0/1 ────────────────────────────────
# 67B in bfloat16 ~134GB; needs 2 GPUs minimum. Using GPU 2 alone won't fit.
# We use GPUs 1+2 at 45% each (second half of GPU 1's remaining capacity).
CUDA_VISIBLE_DEVICES=1,2 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/deepseek-llm-67b-chat \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.45 \
        --port 8005 &
PIDS+=($!)

wait_for_port 8005

# ── 3. Qwen2.5-72B — all 3 GPUs, 90% util ────────────────────────────────────
# Runs after the two above are loaded; 72B in bfloat16 ~144GB needs 3×H100.
CUDA_VISIBLE_DEVICES=0,1,2 apptainer run --nv \
    --bind "$MODEL_DIR":/models \
    "$SIF" \
    vllm serve /models/Qwen2.5-72B-Instruct \
        --tensor-parallel-size 3 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --port 8006 &
PIDS+=($!)

wait_for_port 8006

# ── SSH tunnels ────────────────────────────────────────────────────────────────
for PORT in 8004 8005 8006; do
    ssh -N -o StrictHostKeyChecking=no -R "$PORT:localhost:$PORT" fep &
done

echo ""
echo "=== Job B ready on $(hostname) ==="
echo "  8004 → Mixtral-8x22B-Instruct-v0.1"
echo "  8005 → deepseek-llm-67b-chat"
echo "  8006 → Qwen2.5-72B-Instruct"

wait "${PIDS[@]}"