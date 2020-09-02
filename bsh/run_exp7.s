#!/bin/bash
#
#SBATCH --job-name=exp7
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp7.out
#SBATCH --error=exp7.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python 3_training_nophase.py --save_key exp7multif0 --data_splits_file data_splits.json

