#!/bin/sh
#SBATCH --time=00:15:00         
#SBATCH --nodes=1              

module load vivado/2024.1

vitis_hls -f run_blockmatmul_hls.tcl
