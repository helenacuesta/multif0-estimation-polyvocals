#!/bin/bash
#
#SBATCH --job-name=exp5
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp5.out
#SBATCH --error=exp5.err

module purge

module load cuda/10.1.105
module load cudnn/7.0v4.0


python 2_training.py --model model5 --save_key exp5_multif0
