#!/bin/bash
#
#SBATCH --job-name=data_prep_v0
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=data_prep_v0.out
#SBATCH --error=data_prep_v0.err


module purge
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2


python data_augmentation.py --f0-path /scratch/hc2945/multif0/CSD/individuals
echo CSD data ready!
python data_augmentation.py --f0-path /scratch/hc2945/multif0/ECS/individuals
echo ECS data ready!
python data_augmentation.py --f0-path /scratch/hc2945/multif0/DCS/individuals
echo DCS data ready!