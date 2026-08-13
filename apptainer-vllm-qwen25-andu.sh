#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH -A student
#SBATCH -p dgxa100
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-qwen-%j.log

MODEL_DIR=$HOME/ADI/licenta/models
mkdir -p $MODEL_DIR

SIF=$HOME/ADI/licenta/vllm-ngc.sif

PORT_QWEN=8002

MODEL_ID_QWEN="Qwen/Qwen2.5-32B-Instruct"
MODEL_PATH_QWEN=$MODEL_DIR/Qwen2.5-32B-Instruct

# --- Download models if needed ---

if [ ! -d "$MODEL_PATH_QWEN" ]; then
    echo "Downloading Qwen2.5-32B-Instruct..."
    apptainer exec --nv \
        --bind $MODEL_DIR:/models \
        $SIF \
        huggingface-cli download "$MODEL_ID_QWEN" \
            --local-dir /models/Qwen2.5-32B-Instruct
fi

# --- Launch Qwen2.5-32B-Instruct on GPU 2 only ---
# Note: 27B in bf16 ~54GB, fits on a single H100 80GB but context is limited.
# Reduce --max-model-len if vLLM complains about KV cache OOM.

echo "Starting Qwen2.5-32B-Instruct on port $PORT_QWEN (GPU 2)..."
CUDA_VISIBLE_DEVICES=0,1 apptainer run --nv \
    --bind $MODEL_DIR:/models \
    $SIF \
    vllm serve /models/Qwen2.5-32B-Instruct \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.90 \
        --port $PORT_QWEN &

QWEN_PID=$!

echo "Waiting for Qwen server..."
until curl -sf http://localhost:$PORT_QWEN/health > /dev/null 2>&1; do sleep 2; done
echo "Qwen is up!"

# --- SSH tunnels ---

ssh -N -o StrictHostKeyChecking=no -i $HOME/.ssh/id_andu -R $PORT_QWEN:localhost:$PORT_QWEN fep &

echo ""
echo "=== Both servers ready! ==="
echo "On your local machine run:"
echo "  ssh -L $PORT_QWEN:localhost:$PORT_QWEN adrian.ariton@fep.grid.pub.ro"
echo ""
echo "Qwen2.5-32B-Instruct → port $PORT_QWEN (GPU 2)"
echo "  curl http://localhost:$PORT_QWEN/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/models/Qwen2.5-32B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"

wait $QWEN_PID