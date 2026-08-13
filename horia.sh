#!/bin/bash
#SBATCH --job-name=horia
#SBATCH -A student
#SBATCH -p dgxh100
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=16G
#SBATCH --output=horia-%j.log

curl -X POST http://dgxh100-precis-wn02:8001/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{
  "model": "/models/Qwen3-Next-80B-A3B-Instruct-FP8",
  "messages": [{"role": "user", "content": "Hello"}]
}'