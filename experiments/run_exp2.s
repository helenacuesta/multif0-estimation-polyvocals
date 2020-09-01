#!/bin/bash
#
#SBATCH --job-name=exp2
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp2.out
#SBATCH --error=exp2.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python 2_training.py --model model2 --save_key exp2multif0 --data_splits_file data_splits.json

