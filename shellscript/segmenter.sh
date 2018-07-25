#!/bin/bash

#SBATCH -c 2
#SBATCH -t 0-23:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -o segmenter%j.out
#SBATCH -e segmenter%j.err

#python run_full_segmenter.py -e finalsegmenter_sgd1 -o sgd -l 0.01 -n 200
#python run_full_segmenter.py -e finalsegmenter_sgd2 -o sgd -l 0.005 -n 200
#python run_full_segmenter.py -e finalsegmenter_sgd3 -o sgd -l 0.001 -n 200
#python run_full_segmenter.py -e finalsegmenter_sgd4 -o sgd -l 0.0005 -n 200
#python run_full_segmenter.py -e finalsegmenter_sgd5 -o sgd -l 0.0001 -n 200


#python run_full_segmenter.py -e finalsegmenter_adam1 -o adam -l 0.01 -n 200
#python run_full_segmenter.py -e finalsegmenter_adam2 -o adam -l 0.005 -n 200
#python run_full_segmenter.py -e finalsegmenter_adam3 -o adam -l 0.001 -n 200
#python run_full_segmenter.py -e finalsegmenter_adam4 -o adam -l 0.0005 -n 200
#python run_full_segmenter.py -e finalsegmenter_adam5 -o adam -l 0.0001 -n 200

#python run_full_segmenter.py -e finalsegmenter_adam21 -o adam -l 0.005 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam22 -o adam -l 0.004 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam23 -o adam -l 0.003 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam24 -o adam -l 0.002 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam25 -o adam -l 0.001 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam26 -o adam -l 0.0009 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam27 -o adam -l 0.0008 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam28 -o adam -l 0.0007 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam29 -o adam -l 0.0006 -n 300
#python run_full_segmenter.py -e finalsegmenter_adam30 -o adam -l 0.0005 -n 300
#python run_full_segmenter.py -e reduced_segmenter1 -o adam -l 0.005 -n 100 -t 0.4
#python run_full_segmenter.py -e reduced_segmenter2 -o adam -l 0.005 -n 100 -t 0.6
#python run_full_segmenter.py -e reduced_segmenter3 -o adam -l 0.005 -n 100 -t 0.8


#python run_full_segmenter.py -e reduced_segmenter_adam0 -o adam -l 0.002 -n 500 -t 1.0
#python run_full_segmenter.py -e reduced_segmenter_adam1 -o adam -l 0.002 -n 500 -t 0.9
#python run_full_segmenter.py -e reduced_segmenter_adam2 -o adam -l 0.002 -n 500 -t 0.8
#python run_full_segmenter.py -e reduced_segmenter_adam3 -o adam -l 0.002 -n 500 -t 0.7
#python run_full_segmenter.py -e reduced_segmenter_adam4 -o adam -l 0.002 -n 500 -t 0.6
#python run_full_segmenter.py -e reduced_segmenter_adam5 -o adam -l 0.002 -n 500 -t 0.5
#python run_full_segmenter.py -e reduced_segmenter_adam6 -o adam -l 0.002 -n 500 -t 0.4
#python run_full_segmenter.py -e reduced_segmenter_adam7 -o adam -l 0.002 -n 500 -t 0.3
#python run_full_segmenter.py -e reduced_segmenter_adam8 -o adam -l 0.002 -n 500 -t 0.2
#python run_full_segmenter.py -e reduced_segmenter_adam9 -o adam -l 0.002 -n 500 -t 0.1


#python run_full_segmenter.py -e reduced_segmenter_sgd00 -o sgd -l 0.005 -n 500 -t 1.0
#python run_full_segmenter.py -e reduced_segmenter_sgd01 -o sgd -l 0.005 -n 500 -t 0.9
#python run_full_segmenter.py -e reduced_segmenter_sgd02 -o sgd -l 0.005 -n 500 -t 0.8
#python run_full_segmenter.py -e reduced_segmenter_sgd03 -o sgd -l 0.005 -n 500 -t 0.7
#python run_full_segmenter.py -e reduced_segmenter_sgd04 -o sgd -l 0.005 -n 500 -t 0.6
#python run_full_segmenter.py -e reduced_segmenter_sgd05 -o sgd -l 0.005 -n 500 -t 0.5
#python run_full_segmenter.py -e reduced_segmenter_sgd06 -o sgd -l 0.005 -n 500 -t 0.4
#python run_full_segmenter.py -e reduced_segmenter_sgd07 -o sgd -l 0.005 -n 500 -t 0.3
#python run_full_segmenter.py -e reduced_segmenter_sgd08 -o sgd -l 0.005 -n 500 -t 0.2
#python run_full_segmenter.py -e reduced_segmenter_sgd09 -o sgd -l 0.005 -n 500 -t 0.1


#python run_full_segmenter.py -e reduced_segmenter_sgd10 -o sgd -l 0.01 -n 500 -t 1.0
#python run_full_segmenter.py -e reduced_segmenter_sgd11 -o sgd -l 0.01 -n 500 -t 0.9
#python run_full_segmenter.py -e reduced_segmenter_sgd12 -o sgd -l 0.01 -n 500 -t 0.8
#python run_full_segmenter.py -e reduced_segmenter_sgd13 -o sgd -l 0.01 -n 500 -t 0.7
#python run_full_segmenter.py -e reduced_segmenter_sgd14 -o sgd -l 0.01 -n 500 -t 0.6
#python run_full_segmenter.py -e reduced_segmenter_sgd15 -o sgd -l 0.01 -n 500 -t 0.5
#python run_full_segmenter.py -e reduced_segmenter_sgd16 -o sgd -l 0.01 -n 500 -t 0.4
#python run_full_segmenter.py -e reduced_segmenter_sgd17 -o sgd -l 0.01 -n 500 -t 0.3
#python run_full_segmenter.py -e reduced_segmenter_sgd18 -o sgd -l 0.01 -n 500 -t 0.2
#python run_full_segmenter.py -e reduced_segmenter_sgd19 -o sgd -l 0.01 -n 500 -t 0.1


#python run_full_segmenter.py -e aug_segmenter_adam1 -o adam -l 0.002 -n 200 -t 0.2 -a 2
#python run_full_segmenter.py -e aug_segmenter_adam2 -o adam -l 0.002 -n 200 -t 0.2 -a 4
#python run_full_segmenter.py -e aug_segmenter_adam3 -o adam -l 0.002 -n 200 -t 0.2 -a 6
#python run_full_segmenter.py -e aug_segmenter_adam4 -o adam -l 0.002 -n 200 -t 0.2 -a 8
#python run_full_segmenter.py -e aug_segmenter_adam5 -o adam -l 0.002 -n 200 -t 0.2 -a 10



#python run_cv_segmenter.py -e cv_reduced_segmenter_adam0 -o adam -l 0.002 -n 200 -t 1.0 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam1 -o adam -l 0.002 -n 200 -t 0.9 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam2 -o adam -l 0.002 -n 200 -t 0.8 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam3 -o adam -l 0.002 -n 200 -t 0.7 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam4 -o adam -l 0.002 -n 200 -t 0.6 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam5 -o adam -l 0.002 -n 200 -t 0.5 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam6 -o adam -l 0.002 -n 200 -t 0.4 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam7 -o adam -l 0.002 -n 200 -t 0.3 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam8 -o adam -l 0.002 -n 200 -t 0.2 -f 5
#python run_cv_segmenter.py -e cv_reduced_segmenter_adam9 -o adam -l 0.002 -n 200 -t 0.1 -f 5

#python run_cv_segmenter.py -e cv_aug_segmenter_adam0 -o adam -l 0.002 -n 200 -t 1.0 -f 5 -a 10
#python run_cv_segmenter.py -e cv_aug_segmenter_adam1 -o adam -l 0.002 -n 200 -t 1.0 -f 5 -a 8
#python run_cv_segmenter.py -e cv_aug_segmenter_adam2 -o adam -l 0.002 -n 200 -t 1.0 -f 5 -a 6
#python run_cv_segmenter.py -e cv_aug_segmenter_adam3 -o adam -l 0.002 -n 200 -t 1.0 -f 5 -a 4
#python run_cv_segmenter.py -e cv_aug_segmenter_adam4 -o adam -l 0.002 -n 200 -t 1.0 -f 5 -a 2
#python run_cv_segmenter.py -e cv_aug_segmenter_adam_test -o adam -l 0.002 -n 3 -t 1.0 -f 3 -a 2
#python run_cv_segmenter.py -e cv_aug_segmenter_adam_best -o adam -l 0.005 -n 300 -t 1.0 -f 5 -a 5



#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam0 -o adam -l 0.002 -n 200 -t 1.0 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam1 -o adam -l 0.002 -n 200 -t 0.9 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam2 -o adam -l 0.002 -n 200 -t 0.8 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam3 -o adam -l 0.002 -n 200 -t 0.7 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam4 -o adam -l 0.002 -n 200 -t 0.6 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam5 -o adam -l 0.002 -n 200 -t 0.5 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam6 -o adam -l 0.002 -n 200 -t 0.4 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam7 -o adam -l 0.002 -n 200 -t 0.3 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam8 -o adam -l 0.002 -n 200 -t 0.2 -m trainable_segmenter
#python run_cv_all.py -e cv_reduced_trainable_segmenter_adam9 -o adam -l 0.002 -n 200 -t 0.1 -m trainable_segmenter

#python run_cv_segmenter.py -e reduced_segmenter_adam0_test_fullloc -o adam -l 0.002 -n 500 -t 1.0 -f 5
python run_cv_segmenter.py -e cv_aug_segmenter_adam0_test_fullloc -o adam -l 0.002 -n 200 -a 5 -f 5
