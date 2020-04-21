#!/bin/bash
#
#SBATCH --job-name=exp1
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp1.out
#SBATCH --error=exp1.err

module purge

module load cuda/10.1.105
module load cudnn/7.0v4.0


python 2_training.py --model model1 --save_key exp1multif0