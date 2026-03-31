#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=TitanX

module load cuda12.6/toolkit

./TiledMatrixMul_exercise 1024
