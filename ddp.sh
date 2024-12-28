#!/usr/bin/bash

#SBATCH -J ME_ENSEMBLE_DDP
#SBATCH --nodes=1                      # 1개 노드 사용
#SBATCH --ntasks-per-node=4           # 노드당 4개 프로세스
#SBATCH --gres=gpu:4                  # 4개 GPU 요청
#SBATCH --cpus-per-task=8             # 프로세스당 8개 CPU
#SBATCH --mem=128G                    # 총 메모리 (32G * 4)
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out
##SBATCH -w moana-r4                  # 특정 노드 지정 (필요시 주석 해제)

# DDP 환경 변수 설정
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# 현재 환경 정보 출력
which python
hostname
nvidia-smi

# DDP로 학습 실행
srun python ddp_me_train.py

exit 0