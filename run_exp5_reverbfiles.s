#!/bin/bash
#
#SBATCH --job-name=exp5rev
#SBATCH --nodes=1
#SBATCH --time=70:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp5rev.out
#SBATCH --error=exp5rev.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python predict_experiment5.py --model_path /scratch/hc2945/data/models --save_path /scratch/hc2945/data/ChoirDataset/experiment5 --list_of_files ./reverb_files_exp5.txt

