#!/bin/bash
#
#SBATCH --job-name=dstest
#SBATCH --nodes=1
#SBATCH --time=70:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=dstest.out
#SBATCH --error=dstest.err


module purge


module load cudnn/7.0v4.0
module load cuda/10.1.105

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python ds_test_set.py
