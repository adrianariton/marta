#!/bin/bash
#SBATCH --job-name=vllmall
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --output=vllm-multi-%j.log

# --- Path Configuration ---
MODEL_DIR=$HOME/licenta/models
SIF=$HOME/licenta/vllm-ngc.sif
mkdir -p $MODEL_DIR

# --- 1. Automated Model Downloads ---
echo "Checking/Downloading models..."
MODELS=(
    "jondurbin/bagel-34b-v0.2"
    "GraySwanAI/Mistral-7B-Instruct-R"
    "Youliang/llama3-8b-derta"
    "wicai24/Llama-3-8B-Instruct-W-DOOR-exponential"
)

for MODEL_ID in "${MODELS[@]}"; do
    SAFE_NAME=$(echo $MODEL_ID | cut -d'/' -f2)
    MODEL_PATH="$MODEL_DIR/$SAFE_NAME"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Downloading $MODEL_ID..."
        apptainer exec --nv --bind $MODEL_DIR:/models $SIF \
            huggingface-cli download "$MODEL_ID" --local-dir "/models/$SAFE_NAME"
    else
        echo "Model $SAFE_NAME exists."
    fi
done

# --- 2. Launch vLLM Servers ---

# GPU 0: Bagel-34B (Large model, needs full VRAM/Compute)
CUDA_VISIBLE_DEVICES=0 apptainer run --nv --bind $MODEL_DIR:/models $SIF \
    vllm serve /models/bagel-34b-v0.2 \
    --port 8001 --trust-remote-code --dtype bfloat16 \
    --gpu-memory-utilization 0.95 --max-model-len 8192 &
PID1=$!

# GPU 1: Shared by the 3 smaller 7B/8B models
# Using 0.30 utilization each to fit all three on one 80GB card (0.30 * 3 = 0.90)
CUDA_VISIBLE_DEVICES=1 apptainer run --nv --bind $MODEL_DIR:/models $SIF \
    vllm serve /models/Mistral-7B-Instruct-R \
    --port 8002 --dtype bfloat16 \
    --gpu-memory-utilization 0.30 --max-model-len 8192 &
PID2=$!

CUDA_VISIBLE_DEVICES=1 apptainer run --nv --bind $MODEL_DIR:/models $SIF \
    vllm serve /models/llama3-8b-derta \
    --port 8003 --dtype bfloat16 \
    --gpu-memory-utilization 0.30 --max-model-len 8192 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 apptainer run --nv --bind $MODEL_DIR:/models $SIF \
    vllm serve /models/Llama-3-8B-Instruct-W-DOOR-exponential \
    --port 8004 --dtype bfloat16 \
    --gpu-memory-utilization 0.30 --max-model-len 8192 &
PID4=$!

# --- 3. Health Checks & Connectivity ---

echo "Waiting for all vLLM servers to initialize..."
for PORT in 8001 8002 8003 8004; do
    until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 10
    done
    echo "Port $PORT is UP."
done

# Establish Reverse Tunnel to fep
ssh -N -o StrictHostKeyChecking=no \
    -R 8001:localhost:8001 \
    -R 8002:localhost:8002 \
    -R 8003:localhost:8003 \
    -R 8004:localhost:8004 fep &
TUNNEL_PID=$!

echo "---------------------------------------------------------------"
echo "Ready! SSH tunnels established to fep for ports 8001-8004"
echo "vLLM is running on: $(hostname)"
echo ""
echo "On your local machine, run this to bridge all ports:"
echo "  ssh -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 -L 8004:localhost:8004 adrian.ariton@fep.grid.pub.ro"
echo ""
echo "API Endpoints (OpenAI-compatible):"
echo "1. Bagel-34B: http://localhost:8001/v1"
echo "2. Mistral-R: http://localhost:8002/v1"
echo "3. Derta:    http://localhost:8003/v1"
echo "4. DOOR:     http://localhost:8004/v1"
echo "---------------------------------------------------------------"

# Keep script alive as long as vLLM is running
wait $PID1 $PID2 $PID3 $PID4 $TUNNEL_PID