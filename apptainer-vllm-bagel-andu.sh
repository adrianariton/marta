#!/bin/bash
#SBATCH --job-name=vllm-bagel-34b
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:2      
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=vllm-bagel-%j.log

MODEL_DIR=$HOME/ADI/licenta/models
mkdir -p $MODEL_DIR

PORT=8001
SIF=$HOME/ADI/licenta/vllm-ngc.sif

# Changed to Bagel 34B v0.2
MODEL_ID="jondurbin/bagel-34b-v0.2"
MODEL_PATH=$MODEL_DIR/bagel-34b-v0.2

if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading model..."
    apptainer exec --nv \
        --bind $MODEL_DIR:/models \
        $SIF \
        huggingface-cli download "$MODEL_ID" \
            --local-dir /models/bagel-34b-v0.2
fi

# Launch vLLM
# Note: Added --trust-remote-code for Yi architecture compatibility
apptainer run --nv \
    --bind $MODEL_DIR:/models \
    $SIF \
    vllm serve /models/bagel-34b-v0.2 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.90 \
        --port $PORT &

VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 2
done
echo "Server is up!"

ssh -N -o StrictHostKeyChecking=no -i $HOME/.ssh/id_andu -R $PORT:localhost:$PORT fep &
TUNNEL_PID=$!

echo "Ready! SSH tunnel established to fep:$PORT"
echo "On your local machine run:"
echo "  ssh -L $PORT:localhost:$PORT adrian.ariton@fep.grid.pub.ro"
echo "vLLM is running on: $(hostname)"
echo ""
echo "API is OpenAI-compatible. Example (Note the model name in the JSON):"
echo "  curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/models/bagel-34b-v0.2\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"

wait $VLLM_PID