#!/bin/bash
#SBATCH -J oavnn
#SBATCH --partition pierson
#SBATCH --output out/oavnn_cap_run0.out
#SBATCH -e out/oavnn_cap_run0.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2377@cornell.edu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --cores-per-socket=4
$@

python run_models.py --exp_name oavnn_cap_run0 --model oavnn --rot so3 --class_choice cap --data_path /home/sb2377/shapenet_single_class --num_points 128
