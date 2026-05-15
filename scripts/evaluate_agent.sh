#!/bin/bash
#SBATCH --job-name=fakenews_eval
#SBATCH --partition=gpu
#SBATCH --gres=shard:16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate llm_env

export CUDA_HOME=/apps/codes/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /home/akumar/fakenews-agent

echo "=== Starting evaluation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Time: $(date)"

python scripts/evaluate_agent.py
echo "=== Done: $(date) ==="
