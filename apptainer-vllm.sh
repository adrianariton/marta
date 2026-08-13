#!/bin/bash
#SBATCH --job-name=vllm-72b
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-%j.log

MODEL_DIR=$HOME/licenta/models
mkdir -p $MODEL_DIR

PORT=8001
SIF=$HOME/licenta/vllm-ngc.sif

MODEL_ID="Qwen/Qwen2.5-72B-Instruct"
MODEL_PATH=$MODEL_DIR/Qwen2.5-72B-Instruct

if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading model..."
    apptainer exec --nv \
        --bind $MODEL_DIR:/models \
        $SIF \
        huggingface-cli download "$MODEL_ID" \
            --local-dir /models/Qwen2.5-72B-Instruct
fi

# Launch vLLM
apptainer run --nv \
    --bind $MODEL_DIR:/models \
    $SIF \
    vllm serve /models/Qwen2.5-72B-Instruct \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.92 \
        --enable-chunked-prefill \
        --max-num-seqs 64 \
        --port $PORT &

VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 2
done
echo "Server is up!"

ssh -N -o StrictHostKeyChecking=no -R $PORT:localhost:$PORT fep &
TUNNEL_PID=$!

echo "Ready! SSH tunnel established to fep:$PORT"
echo "On your local machine run:"
echo "  ssh -L $PORT:localhost:$PORT adrian.ariton@fep.grid.pub.ro"
echo "vLLM is running on: $(hostname)"
echo ""
echo "API is OpenAI-compatible. Example:"
echo "  curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/models/Qwen2.5-72B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"

wait $VLLM_PID