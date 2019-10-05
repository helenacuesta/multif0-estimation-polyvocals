#!/bin/bash
#
#SBATCH --job-name=data_prep_v0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=setup.out
#SBATCH --error=setup.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python 0_setup.py