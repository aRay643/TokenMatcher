#!/usr/bin/bash
#SBATCH --partition=gpu       # ָ������GPU����
#SBATCH --account=chenjun3    # ָ���˻���
#SBATCH --ntasks=1            # ָ��������������Ϊ1������
#SBATCH --ntasks-per-node=4   # ÿ���ڵ��������������Ϊ4
#SBATCH --cpus-per-task=5     # ÿ�������CPU������������Ϊ5
#SBATCH --gres=gpu:4          # ָ��GPU��Դ��������Ϊ4��GPU
#SBATCH --exclude=g0045,g0017,g0015  # �ų��Ľڵ�
#SBATCH -o test_regdb_vis_query.log    # �����־�ļ�
# module load scl/gcc5.3      # ����ָ��������ģ�飨�����ﱻע�͵��ˣ�


module load nvidia/cuda/11.6

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_regdb_vis_query.py -b 128 -a vit_base -d regdb_rgb --iters 100 --num-instances 16 --self-norm --hw-ratio 2 --conv-stem \
-pp /scratch/chenjun3/liulekai/PGM-ReID-main/examples/pretrained/vit_base_ics_cfs_lup.pth --cls-token-num 6
 
#-m torch.distributed.launch --nproc_per_node=4