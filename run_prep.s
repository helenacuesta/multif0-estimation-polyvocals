#!/bin/bash
#
#SBATCH --job-name=prep
#SBATCH --njobs=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --output=prep.out
#SBATCH --error=prep.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python 1_prep.py --audio-path /scratch/hc2945/multif0/AudioMixtures --metadata-path /scratch/hc2945/multif0/AudioMixtures/mtracks_info.json --save-dir /scratch/hc2945/multif0/AudioMixtures