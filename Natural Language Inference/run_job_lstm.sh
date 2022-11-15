#!/bin/sh
#SBATCH --job-name=mainak_lstm_test
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=mainak_lstm_test%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=all_srv2
#SBATCH --error=mainak_lstm_test%j.err

export CUDA_LAUNCH_BLOCKING=1
python /home/mainakbiswas/Project_final_mainak/driver_code_lstm.py
