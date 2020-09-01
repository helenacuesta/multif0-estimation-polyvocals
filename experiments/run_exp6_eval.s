#!/bin/bash
#
#SBATCH --job-name=eval6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem=16GB
#SBATCH --output=eval6.out
#SBATCH --error=eval6.err


module purge

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python eval_exp6_nophase.py
