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

PORT=8002
SIF=$HOME/licenta/vllm-ngc.sif

MODEL_ID="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_PATH=$MODEL_DIR/Qwen3-Next-80B-A3B-Instruct-FP8

if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading FP8 model..."
    apptainer exec --nv \
        --bind $MODEL_DIR:/models \
        $SIF \
        huggingface-cli download "$MODEL_ID" \
            --local-dir /models/Qwen3-Next-80B-A3B-Instruct-FP8
fi

# Launch vLLM
# Create the writable flashinfer dir on the host first
mkdir -p $HOME/.cache/flashinfer-csrc

apptainer run --nv \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env VLLM_USE_TRITON_FLASH_ATTN=0 \
    --bind $MODEL_DIR:/models \
    --bind $HOME/.cache/flashinfer-csrc:/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/cutlass_instantiations \
    $SIF \
    vllm serve /models/Qwen3-Next-80B-A3B-Instruct-FP8 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.95 \
        --enable-chunked-prefill \
        --max-num-seqs 32 \
        --disable-custom-all-reduce \
        --port $PORT &
VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 2
done
echo "Server is up!"

ssh -N -o -i  StrictHostKeyChecking=no -R $PORT:localhost:$PORT fep &
TUNNEL_PID=$!

echo "Ready! SSH tunnel established to fep:$PORT"
echo "On your local machine run:"
echo "  ssh -L $PORT:localhost:$PORT adrian.ariton@fep.grid.pub.ro"
echo "vLLM is running on: $(hostname)"
echo ""
echo "API is OpenAI-compatible. Example:"
echo "  curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/models/Qwen3-Next-80B-A3B-Instruct-FP8\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"

wait $VLLM_PID