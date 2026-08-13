#!/bin/bash
#SBATCH --job-name=cb-ft-eval
#SBATCH -A student
#SBATCH -p dgxa100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --output=log-cb-eval-%j.log

# ---------------------------------------------------------------------------
# Usage:
#   sbatch apptainer-eval-cb.sh            # auto-resume from latest checkpoint
#   sbatch apptainer-eval-cb.sh --force    # start from scratch
# ---------------------------------------------------------------------------

FORCE_FLAG=""
for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE_FLAG="--force"
done

# Dynamically pick a random port between 15000 and 25000 to avoid collisions
export MASTER_PORT=$(shuf -i 15000-25000 -n 1)

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Master Port:  $MASTER_PORT"
echo "Start:        $(date)"
echo "Force:        ${FORCE_FLAG:-no}"
echo "=========================================="

mkdir -p logs

# Apptainer automatically passes host environment variables (like MASTER_PORT) 
# inside the container, which torchrun will pick up.
apptainer exec --nv env.sif \
  torchrun \
    --master_port=$MASTER_PORT \
    --nproc_per_node=1 \
    train/eval_cb.py --config train/config_cb.yaml \
    $FORCE_FLAG

echo "End: $(date)"