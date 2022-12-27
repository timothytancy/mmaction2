#!/usr/bin/env bash
conda init bash
conda activate open-mmlab
python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --work-dir work_dirs/stgcn_80e_ntu60_xsub_keypoint \
    --validate --seed 0 --deterministic \
    --cfg-options gpu_ids="[0, 1, 2, 3]"