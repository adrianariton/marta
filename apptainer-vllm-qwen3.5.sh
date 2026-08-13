#!/bin/bash
#SBATCH --job-name=vllm-qwen35
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-qwen35-%j.log

MODEL_DIR=$HOME/licenta/models
mkdir -p $MODEL_DIR

SIF=$HOME/licenta/vllm-ngc.sif
PORT=8001

MODEL_ID="Qwen/Qwen3.5-27B"
MODEL_PATH=$MODEL_DIR/Qwen3.5-27B

# --- Download model if needed ---

if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading Qwen3.5-27B..."
    apptainer exec --nv \
        --bind $MODEL_DIR:/models \
        $SIF \
        huggingface-cli download "$MODEL_ID" \
            --local-dir /models/Qwen3.5-27B
fi

# --- Launch Qwen3.5-27B on GPUs 0,1 ---
# --reasoning-parser qwen3: enables CoT / thinking mode (<think> tags)
# --max-model-len 32768: more KV cache room now that we have 2 GPUs
# Note: vLLM main branch required for Qwen3.5 support - ensure your .sif is up to date

echo "Starting Qwen3.5-27B on port $PORT (GPUs 0,1)..."
apptainer run --nv \
    --bind $MODEL_DIR:/models \
    $SIF \
    vllm serve /models/Qwen3.5-27B \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser qwen3 \
        --port $PORT &

VLLM_PID=$!

# --- Wait for server ---

echo "Waiting for Qwen server..."
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done
echo "Server is up!"

# --- SSH tunnel ---

ssh -N -o StrictHostKeyChecking=no -R $PORT:localhost:$PORT fep &

echo ""
echo "=== Qwen3.5-27B ready! ==="
echo "On your local machine run:"
echo "  ssh -L $PORT:localhost:$PORT adrian.ariton@fep.grid.pub.ro"
echo ""
echo "API is OpenAI-compatible. CoT thinking is active via --reasoning-parser qwen3."
echo "To trigger thinking mode, set temperature=1.0 and include 'think' in extra_body."
echo ""
echo "Example:"
echo "  curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/models/Qwen3.5-27B\", \"temperature\": 1.0, \"messages\": [{\"role\": \"user\", \"content\": \"Solve step by step: if x^2 - 5x + 6 = 0, find x.\"}]}'"

wait $VLLM_PID