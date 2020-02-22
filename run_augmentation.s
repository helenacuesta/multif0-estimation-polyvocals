#!/bin/bash
#
#SBATCH --job-name=augmentation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=augm.out
#SBATCH --error=augm.err


module purge

module load ffmpeg/intel/3.2.2

python data_augmentation.py --f0-path /scratch/hc2945/data/BachChorales/BC --audio-path /scratch/hc2945/data/BachChorales/BC --pyin yes
python data_augmentation.py --f0-path /scratch/hc2945/data/BarbershopQuartets/BQ --audio-path /scratch/hc2945/data/BarbershopQuartets/BQ --pyin yes
python data_augmentation.py --f0-path /scratch/hc2945/data/CSD --audio-path /scratch/hc2945/data/CSD --pyin no
python data_augmentation.py --f0-path /scratch/hc2945/data/ECS --audio-path /scratch/hc2945/data/ECS --pyin no
python data_augmentation.py --f0-path /scratch/hc2945/data/DCS/annotations_csv_F0_PYIN --audio-path /scratch/hc2945/data/DCS/audio_wav_22050_mono --pyin no


