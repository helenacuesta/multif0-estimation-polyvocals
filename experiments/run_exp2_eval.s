#!/bin/bash
#
#SBATCH --job-name=eval2
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=eval2.out
#SBATCH --error=eval2.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python 3_thresh.py --model model2 --save_key exp2multif0

