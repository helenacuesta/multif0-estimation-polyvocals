#!/bin/bash
#
#SBATCH --job-name=count
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=count.out
#SBATCH --error=count.err


module purge


python count_files.py