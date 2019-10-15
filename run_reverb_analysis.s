#!/bin/bash
#
#SBATCH --job-name=reverb
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=reverb.out
#SBATCH --error=reverb.err

module purge

module load cuda/10.1.105
module load cudnn/7.0v4.0


python predict_and_eval_reverb.py
