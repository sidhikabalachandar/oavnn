#!/bin/bash
#SBATCH --partition orion
#SBATCH --output /sailhome/sidhikab/vnn-articulated/dgcnn/outputs/vnn_128_run0.txt
#SBATCH --gres=gpu:3
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --cores-per-socket=4
$@
