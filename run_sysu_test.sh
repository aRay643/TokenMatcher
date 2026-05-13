#!/usr/bin/bash
#SBATCH --partition=gpu       # ָ������GPU����
#SBATCH --account=chenjun3    # ָ���˻���
#SBATCH --ntasks=1            # ָ��������������Ϊ1������
#SBATCH --ntasks-per-node=4   # ÿ���ڵ��������������Ϊ4
#SBATCH --cpus-per-task=5     # ÿ�������CPU������������Ϊ5
#SBATCH --gres=gpu:4          # ָ��GPU��Դ��������Ϊ4��GPU
#SBATCH --exclude=g0045,g0017,g0015  # �ų��Ľڵ�
#SBATCH -o test_sysu_mix0.4_RI0.1.log    # �����־�ļ�
# module load scl/gcc5.3      # ����ָ��������ģ�飨�����ﱻע�͵��ˣ�

module load nvidia/cuda/11.6
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_sysu.py -b 128 -a vit_base -d sysu_all --num-instances 16 --data-dir "/scratch/chenjun3/liulekai/ADCA/data/sysu" --iters 200 --self-norm --conv-stem \
# -pp /scratch/chenjun3/liulekai/PGM-ReID-main/examples/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 4 --lamba-neighbor 0 --lamba-cross 0 --lamba-mate 0 --lamba-k 2

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_sysu.py -b 128 -a vit_base -d sysu_all --num-instances 16 --data-dir "/scratch/chenjun3/liulekai/ADCA/data/sysu" --iters 200 --self-norm --conv-stem \
-pp /home/lr/code/project/TokenMatcher/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 4 --lamba-neighbor 0 --lamba-cross 0 --lamba-mate 0 --lamba-k 2
#-m torch.distributed.launch --nproc_per_node=4