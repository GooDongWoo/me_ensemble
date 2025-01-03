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
python me_train.py

exit 0
#dos2unix test.sh