#!/bin/bash
#SBATCH --job-name=nemotron-guard-test
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --output=log-nemotron-guard-test-%j.log

# ---------------------------------------------------------------------------
# Usage:
#   sbatch apptainer-test-nemotronguard.sh            # auto-resume from latest checkpoint
#   sbatch apptainer-test-nemotronguard.sh --force    # start from scratch
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
  python train/test_nemotron_and_Adi.py --test ./train/test.parquet --ft_model ./checkpoints/nemotron-guard-ft-weighted-2-1p5/best --batch_size 32 --max_new_tokens 150 --skip_base --out ./eval_results_clean.parquet


echo "End: $(date)"
