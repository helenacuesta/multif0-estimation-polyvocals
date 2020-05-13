#!/bin/bash
#
#SBATCH --job-name=dstest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem=16GB
#SBATCH --output=dstest.out
#SBATCH --error=dstest.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python ds_test_set.py
