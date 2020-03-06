#!/bin/bash
#
#SBATCH --job-name=setup_v0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=setup2.out
#SBATCH --error=setup2.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python 0_setup.py