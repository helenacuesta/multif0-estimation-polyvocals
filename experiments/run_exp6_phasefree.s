#!/bin/bash
#
#SBATCH --job-name=exp6nophase
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp6nophase.out
#SBATCH --error=exp6nophase.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python exp6_test_set_phase_free.py --model_path /scratch/hc2945/data/models --save_path /scratch/hc2945/data/experiment_output/exp6multif0

