#!/bin/bash
#

#SBATCH --job-name=musingersprep
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=musingersprep.out
#SBATCH --error=musingersprep.err

module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python prep_audio_musingers.py
