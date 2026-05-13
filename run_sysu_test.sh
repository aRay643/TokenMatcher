#!/bin/bash
#SBATCH --job-name=tokenmatcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=batch
#SBATCH --output=tm-sysu-test-%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate python3.10

cd /home/lr/code/project/TokenMatcher

CUDA_VISIBLE_DEVICES=0 python test_sysu.py \
  -b 128 \
  -a vit_base \
  -d sysu_all \
  --num-instances 16 \
  --data-dir /home/lr/code/dataset/VI-ReID/SYSU-MM01 \
  --logs-dir /home/lr/code/project/TokenMatcher/logs \
  --iters 200 \
  --self-norm \
  --conv-stem \
  -pp /home/lr/code/project/TokenMatcher/pretrained/vit_base_ics_cfs_lup.pth \
  --cls-token-num 4 \
  --lamba-neighbor 0 \
  --lamba-cross 0 \
  --lamba-mate 0 \
  --lamba-k 2
