#!/bin/bash
#
#SBATCH --job-name=exp4bsq
#SBATCH --nodes=1
#SBATCH --time=98:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp4bsq.out
#SBATCH --error=exp4bsq.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python exp4_bsq.py --model_path /scratch/hc2945/data/models --save_path /scratch/hc2945/data/experiment_output/exp4multif0/bsq --list_of_files ./bsq_list.txt

