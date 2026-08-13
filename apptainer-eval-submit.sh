#!/bin/bash
# Master script — submits 2 SLURM jobs, each serving 3 large models on separate ports.
# Job A: Qwen2.5-32B · Yi-34B-Chat · Llama-3.3-70B
# Job B: Mixtral-8x22B · DeepSeek-67B · Qwen2.5-72B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_A=$(sbatch --parsable "$SCRIPT_DIR/vllm_job_a.sh")
echo "Submitted Job A (qwen2.5-32b / yi-34b / llama-3.3-70b)       → job id $JOB_A"

JOB_B=$(sbatch --parsable "$SCRIPT_DIR/vllm_job_b.sh")
echo "Submitted Job B (mixtral-8x22b / deepseek-67b / qwen2.5-72b)  → job id $JOB_B"

echo ""
echo "Port map:"
echo "  8001 → Qwen2.5-32B-Instruct          (Job A)"
echo "  8002 → Yi-34B-Chat                   (Job A)"
echo "  8003 → Llama-3.3-70B-Instruct        (Job A)"
echo "  8004 → Mixtral-8x22B-Instruct-v0.1   (Job B)"
echo "  8005 → deepseek-llm-67b-chat          (Job B)"
echo "  8006 → Qwen2.5-72B-Instruct           (Job B)"
echo ""
echo "Local SSH tunnel (run once on your laptop):"
echo "  ssh -L 8001:localhost:8001 -L 8002:localhost:8002 -L 8003:localhost:8003 \\"
echo "      -L 8004:localhost:8004 -L 8005:localhost:8005 -L 8006:localhost:8006 \\"
echo "      adrian.ariton@fep.grid.pub.ro"