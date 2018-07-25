#!/bin/bash

#SBATCH -c 2
#SBATCH -t 0-23:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -o compare%j.out
#SBATCH -e compare%j.err


#python run_cv_all.py -e cv_reduced_segmenter2_adam10_test -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam10 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam11 -o adam -l 0.002 -n 500 -t 0.9 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam12 -o adam -l 0.002 -n 500 -t 0.8 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam13 -o adam -l 0.002 -n 500 -t 0.7 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam14 -o adam -l 0.002 -n 500 -t 0.6 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam15 -o adam -l 0.002 -n 500 -t 0.5 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam16 -o adam -l 0.002 -n 500 -t 0.4 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam17 -o adam -l 0.002 -n 500 -t 0.3 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam18 -o adam -l 0.002 -n 500 -t 0.2 -m segmenter -p Full_Localizer1
#python run_cv_all.py -e cv_reduced_segmenter2_adam19 -o adam -l 0.002 -n 500 -t 0.1 -m segmenter -p Full_Localizer1

#python run_cv_all.py -e cv_aug_segmenter2_adam10 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1 -a 2
#python run_cv_all.py -e cv_aug_segmenter2_adam11 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1 -a 3
#python run_cv_all.py -e cv_aug_segmenter2_adam12 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1 -a 4
#python run_cv_all.py -e cv_aug_segmenter2_adam13 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1 -a 5
#python run_cv_all.py -e cv_aug_segmenter2_adam14 -o adam -l 0.002 -n 500 -t 1.0 -m segmenter -p Full_Localizer1 -a 6

python run_cv_all.py -e cv_aug_segmenter2_adam_report -o adam -l 0.002 -n 200 -t 1.0 -m segmenter -p Full_Localizer1 -a 5
