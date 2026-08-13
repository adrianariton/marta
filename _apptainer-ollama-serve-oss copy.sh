#!/bin/bash
#SBATCH --job-name=gpt-oss-120b
#SBATCH -A student
#SBATCH -p dgxa100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=gpt-oss-%j.log

# Use explicit path, no $SCRATCH
OLLAMA_DIR=$HOME/licenta/ollama
mkdir -p $OLLAMA_DIR/models

export OLLAMA_HOST=0.0.0.0:11434

# Bind $HOME/licenta/ollama → /ollama inside container
# and set OLLAMA_MODELS to the container-side path
apptainer run --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --env OLLAMA_HOST=0.0.0.0:11434 \
    --bind $OLLAMA_DIR:/ollama \
    --env OLLAMA_NUM_PARALLEL=4 \
    $HOME/licenta/ollama.sif serve &

echo "Waiting for Ollama server..."
until curl -s http://localhost:11434 > /dev/null 2>&1; do
    sleep 2
done
echo "Server is up!"

apptainer exec --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --bind $OLLAMA_DIR:/ollama \
    $HOME/licenta/ollama.sif ollama pull gpt-oss:120b
apptainer exec --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --bind $OLLAMA_DIR:/ollama \
    $HOME/licenta/ollama.sif ollama list | grep -q "llama3.3" || \
apptainer exec --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --bind $OLLAMA_DIR:/ollama \
    $HOME/licenta/ollama.sif ollama pull llama3.3:70b
apptainer exec --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --bind $OLLAMA_DIR:/ollama \
    $HOME/licenta/ollama.sif ollama list | grep -q "dolphin-llama3" || \
apptainer exec --nv \
    --env OLLAMA_MODELS=/ollama/models \
    --bind $OLLAMA_DIR:/ollama \
    $HOME/licenta/ollama.sif ollama pull dolphin-llama3:70b
ssh -N -o StrictHostKeyChecking=no -R 11434:localhost:11434 fep &
TUNNEL_PID=$!

echo "Ready! SSH tunnel established to fep:11434"
echo "On your local machine run:"
echo "  ssh -L 11434:localhost:11434 adrian.ariton@fep.grid.pub.ro"
echo "Ollama is running on: $(hostname)"
wait