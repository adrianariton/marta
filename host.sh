#!/bin/bash
#SBATCH --job-name=agent-host
#SBATCH -A student
#SBATCH -p dgxa100
#SBATCH --gres=gpu:3  
#SBATCH --output=agent-host_%j.log
#SBATCH --error=agent-host_%j.log
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G

# ---- config ----
PORT=8004
FEP_HOST=fep   # change if your fep alias/hostname differs
SIF=env.sif

# ---- open reverse tunnel: fep:$PORT -> this node:$PORT ----
ssh -N -o StrictHostKeyChecking=no -R ${PORT}:localhost:${PORT} ${FEP_HOST} &
TUNNEL_PID=$!
echo "[slurm] reverse tunnel pid=${TUNNEL_PID} -> ${FEP_HOST}:${PORT}"

cleanup() {
    echo "[slurm] cleaning up..."
    kill "${TUNNEL_PID}" 2>/dev/null
}
trap cleanup EXIT

# ---- ensure flask + flask-cors are available inside the sif ----
# --user installs land in $HOME/.local, which singularity mounts by
# default, so this only needs to run once (cached across jobs).
apptainer exec --nv "${SIF}" \
    python -c "import flask, flask_cors" 2>/dev/null \
    || apptainer exec --nv "${SIF}" \
       pip install --user flask flask-cors

# ---- verify GPU visibility ----
apptainer exec --nv "${SIF}" \
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); [print(f'GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

# ---- run the server (foreground) ----
singularity exec --nv "${SIF}" python host.py