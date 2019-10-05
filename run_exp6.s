#!/bin/bash
#
#SBATCH --job-name=exp6
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp6.out
#SBATCH --error=exp6.err

module purge

module load cuda/10.1.105
module load cudnn/7.0v4.0


python 2_training.py --model model6 --save_key exp6_multif0
