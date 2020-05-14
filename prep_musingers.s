#!/bin/bash
#
#SBATCH --job-name=musingersprep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem=16GB
#SBATCH --output=musingersprep.out
#SBATCH --error=musingersprep.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python prep_audio_musingers.py
