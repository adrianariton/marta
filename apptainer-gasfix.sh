#!/bin/bash
#SBATCH --job-name=gas-fix
#SBATCH -A student
#SBATCH -p dgxa100
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=gas-fix-%j.log

apptainer exec --nv env.sif python -c "
import pandas as pd
df = pd.read_parquet('./eval_results.parquet')
print('pred_ft dtype:', df['pred_ft'].dtype)
print(df.columns)
print('sample values:')
for v in df['pred_base'].dropna().head(10):
    print(repr(v))
"