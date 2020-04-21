#!/bin/bash
#
#SBATCH --job-name=groundtruth
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=gt.out
#SBATCH --error=gt.err


module purge


python compute_multif0_from_activation.py
