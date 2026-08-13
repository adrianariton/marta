#!/bin/bash
#SBATCH --job-name=cb-yeet
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --output=log-cb-yeet-%j.log
apptainer exec --nv env.sif python ~/licenta/test_model.py $@
