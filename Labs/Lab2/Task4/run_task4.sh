#!/bin/bash
#SBATCH --job-name=task4
#SBATCH --output=task4_output_%j.txt
#SBATCH --error=task4_error_%j.txt
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

echo "=========================================="
echo " LAB 2 - TASK 4: Grid-Stride Loop"
echo "=========================================="
echo ""

echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run
echo "Running ..."
echo "------------------------------------------"
./task4
echo "------------------------------------------"
echo "Done."
