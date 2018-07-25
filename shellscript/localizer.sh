#!/bin/bash

#SBATCH -c 2
#SBATCH -t 0-48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -o localizer%j.out
#SBATCH -e localizer%j.err

python run_loc_net.py -e Full_localizer2 -l 0.005 -o rmsprop -n 1000 -t 1
