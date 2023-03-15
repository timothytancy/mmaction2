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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python tools/test.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    work_dirs/default/best_top1_acc_epoch_80.pth --eval top_k_accuracy mean_class_accuracy \
    --out results/default.pkl

    