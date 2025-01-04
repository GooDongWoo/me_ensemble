#!/usr/bin/bash

#SBATCH -J ME_ENSEMBLE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out
#SBATCH -w moana-r4

which python
hostname

#python backbone_train.py
#python me_train.py -u "[0,1,2,3,4,5,6,7,8,9]"
#python integrate_ee.py
#python me_test.py
#python make_last_cache.py
python matrix_scaling.py
python temperature_scaling.py

exit 0
#dos2unix test.sh
