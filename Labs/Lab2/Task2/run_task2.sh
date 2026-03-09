#!/bin/bash
#SBATCH --job-name=task2_matmul
#SBATCH --output=task2_output_%j.txt
#SBATCH --error=task2_error_%j.txt
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

echo "=========================================="
echo " LAB 2 - TASK 2: Implement the Kernel"
echo "=========================================="
echo ""

# Load CUDA module
module load cuda12.3/toolkit/12.3

echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Compile
echo "Compiling task2_matmul.cu ..."
nvcc task2_matmul.cu -o task2_matmul -arch=sm_52
if [ $? -ne 0 ]; then
    echo "Compilation FAILED"
    exit 1
fi
echo "Compilation OK"
echo ""

# Run
echo "Running ..."
echo "------------------------------------------"
./task2_matmul
echo "------------------------------------------"
echo "Done."
