#!/bin/bash
#SBATCH --job-name=test_hf
#SBATCH --partition=xl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=test_mistral.log

# ----------------------------
# Activează virtualenv-ul Python 3.12
# ----------------------------
source ~/licenta/venv312c/bin/activate

# ----------------------------
# Rulează scriptul tău Python
# ----------------------------
python ~/licenta/test.py
