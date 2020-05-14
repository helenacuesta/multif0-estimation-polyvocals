#!/bin/bash
#
#SBATCH --job-name=eval4
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=eval4.out
#SBATCH --error=eval4.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python 3_thresh.py --model model3 --save_key exp4multif0 --data_splits_file data_splits_exp4.json
