#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20


#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=STGCN_Train
#SBATCH --output=output_%x_%j.out 
#SBATCH --error=error_%x_%j.err

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --validate --seed 0 --deterministic 
    # --cfg-options gpu_ids="[0, 1, 2, 3]"