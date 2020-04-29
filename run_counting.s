#!/bin/bash
#
#SBATCH --job-name=count
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=count.out
#SBATCH --error=count.err


module purge
module load cudnn/7.0v4.0
module load cuda/10.1.105

python count_files.py
