#!/bin/bash
#
#SBATCH --job-name=setup_v2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00
#SBATCH --output=setup3.out
#SBATCH --error=setup3.err


module purge
module load cudnn/7.0v4.0
module load cuda/10.1.105
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2
module load sox/intel/14.4.2

python 0_setup.py
