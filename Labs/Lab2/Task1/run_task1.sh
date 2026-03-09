#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq
#SBATCH --constraint="gpunode,TitanX"
#SBATCH --gres=gpu:1

echo "=========================================="
echo " LAB 2 - TASK 1: Memory Management"
echo "=========================================="
echo ""

# Load CUDA module
module load cuda12.3/toolkit/12.3

echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Compile
echo "Compiling task1.cu ..."
nvcc task1.cu -o task1 -arch=sm_52
if [ $? -ne 0 ]; then
    echo "Compilation FAILED"
    exit 1
fi
echo "Compilation OK"
echo ""

# Run
echo "Running ..."
echo "------------------------------------------"
./task1
echo "------------------------------------------"
echo "Done."
