#!/bin/bash
#
#SBATCH --job-name=setup_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --output=setup3.out
#SBATCH --error=setup3.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2
module load sox/intel/14.4.2

python 0_setup_fast.py
