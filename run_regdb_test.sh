#!/bin/bash
#SBATCH --job-name=tokenmatcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=batch
#SBATCH --output=tm-regdb-test-%j.out


source ~/anaconda3/etc/profile.d/conda.sh
conda activate python3.10

cd /home/lr/code/project/TokenMatcher

CUDA_VISIBLE_DEVICES=0 python test_regdb.py -b 128 -a vit_base -d regdb_rgb --iters 100 --num-instances 16 --self-norm --hw-ratio 2 --conv-stem \
-pp /home/lr/code/project/TokenMatcher/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 6
 
#-m torch.distributed.launch --nproc_per_node=4