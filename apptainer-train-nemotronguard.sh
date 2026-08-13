#!/bin/bash
#SBATCH --job-name=nemotron-guard-ft
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --output=log-nemotron-guard-%j.log

# ---------------------------------------------------------------------------
# Usage:
#   sbatch apptainer-train-nemotronguard.sh            # auto-resume from latest checkpoint
#   sbatch apptainer-train-nemotronguard.sh --force    # start from scratch
# ---------------------------------------------------------------------------

FORCE_FLAG=""
for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE_FLAG="--force"
done

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Start:    $(date)"
echo "Force:    ${FORCE_FLAG:-no}"
echo "=========================================="

mkdir -p logs

apptainer exec --nv env.sif \
  torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    train/train_nemotron_guard.py \
    --config train/config.yaml \
    $FORCE_FLAG

echo "End: $(date)"
