#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=q_ug48
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40


#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=STGCN_Train_nodetach
#SBATCH --output=output_%x_%j.out 
#SBATCH --error=error_%x_%j.err

conda init bash
conda activate open-mmlab
python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --work-dir work_dirs/burn2_temp2_b6_g6 \
    --validate --seed 0 --deterministic \
    --cfg-options gpu_ids="[0, 1, 2, 3]"