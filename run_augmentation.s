#!/bin/bash
#
#SBATCH --job-name=augmentation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=augm.out
#SBATCH --error=augm.err


module purge

module load cudnn/7.0v4.0
module load cuda/10.1.105

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python data_augmentation.py --f0-path /scratch/hc2945/data/BachChorales/BC/pyin_annot --audio-path /scratch/hc2945/data/BachChorales/BC --dataset BC
echo BC done!

python data_augmentation.py --f0-path /scratch/hc2945/data/BarbershopQuartets/BQ/pyin_annot --audio-path /scratch/hc2945/data/BarbershopQuartets/BQ --dataset BSQ
echo BSQ done!

#python data_augmentation.py --f0-path /scratch/hc2945/data/CSD --audio-path /scratch/hc2945/data/CSD --dataset CSD
#echo CSD done!

#python data_augmentation.py --f0-path /scratch/hc2945/data/ECS --audio-path /scratch/hc2945/data/ECS --dataset ECS
#echo ECS done!

python data_augmentation.py --f0-path /scratch/hc2945/data/DCS/annotations_csv_F0_PYIN --audio-path /scratch/hc2945/data/DCS/audio_wav_22050_mono --dataset DCS
echo DCS done!