#!/bin/bash
#
#SBATCH --job-name=exp7
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp7.out
#SBATCH --error=exp7.err

module purge

module load cuda/10.1.105
module load cudnn/7.0v4.0


python 2_training.py --model model7 --save_key exp7_multif0
