#!/bin/sh
#SBATCH --job-name=mainak_RNN_test
#SBATCH --ntasks=1
#SBATCH --time=unlimited
#SBATCH --output=mainak_RNN_test%j.out
#SBATCH --gres=gpu:3
#SBATCH --partition=all_srv2
#SBATCH --error=mainak_RNN_test%j.err

python /home/mainakbiswas/driver_code.py
