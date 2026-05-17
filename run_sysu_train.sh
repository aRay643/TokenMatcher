#!/bin/bash
#SBATCH --job-name=tokenmatcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=batch
#SBATCH --output=tm-train-%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate python3.10

CUDA_VISIBLE_DEVICES=0 python train_sysu.py -b 128 -a vit_base -d sysu_all --num-instances 16 --data-dir "/home/lr/code/dataset/VI-ReID/SYSU-MM01" --iters 200 --self-norm --conv-stem \
-pp /home/lr/code/project/TokenMatcher/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 4 --lamba-cross 0.4 --lamba-neighbor 0.5 --lamba-mate 0.03 --epochs 50

# First, run the first stage of train_sysu to obtain the baseline, then run train_sysu_IRcam and train_sysu_RGBcam to obtain reliable local clusters, 
# and finally, run the second stage of train_sysu.