#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=q_ug48
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40


#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=STGCN_Test
#SBATCH --output=output_%x_%j.out 
#SBATCH --error=error_%x_%j.err

conda activate open-mmlab

python tools/test.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    /home/FYP/ttan063/mmaction2/work_dirs/stgcn_80e_ntu60_xsub_keypoint/epoch_50.pth --eval top_k_accuracy mean_class_accuracy \
    --out results/result.pkl --cfg-options gpu_ids="[0, 1, 2, 3]"

    