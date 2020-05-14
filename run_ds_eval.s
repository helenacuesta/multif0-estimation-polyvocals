#!/bin/bash
#
#SBATCH --job-name=dseval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem=16GB
#SBATCH --output=dseval.out
#SBATCH --error=dseval.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python eval_deepsalience.py
