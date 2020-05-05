#!/bin/bash
#
#SBATCH --job-name=exp3
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp3.out
#SBATCH --error=exp3.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python 2_training.py --model model3 --save_key exp3multif0

