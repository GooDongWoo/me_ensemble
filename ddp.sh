#!/usr/bin/bash

#SBATCH -J ME_ENSEMBLE_DDP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out
#SBATCH -w moana-r4

# 기본 환경 변수 설정
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 현재 환경 정보 출력
which python
hostname
nvidia-smi

# torchrun으로 학습 실행 (랑데부 엔드포인트를 자동으로 설정)
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$(uuidgen) \
    --rdzv_backend=c10d \
    ddp_me_train.py

exit 0