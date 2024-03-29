#!/bin/bash
#
#SBATCH --job-name=exp5choir
#SBATCH --nodes=1
#SBATCH --time=70:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp5choir.out
#SBATCH --error=exp5choir.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105


python predict_experiment5.py --model_path /scratch/hc2945/data/models --save_path /scratch/hc2945/data/experiment_output/ --list_of_files ./choirdataset_files.txt

