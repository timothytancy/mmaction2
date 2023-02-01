conda activate open-mmlab
python tools/analysis/analyze_logs.py \
plot_curve work_dirs/burn2_temp1_b6_g6/soft_tgt.json \
work_dirs/burn2_temp2_b6_g6/20230131_224816.log.json \
work_dirs/default_stgcn/default.json \
--keys top1_acc --legend temp1 temp2 default --iters-per-epoch 627 627 2506 --out top1_acc.pdf

# python tools/analysis/analyze_logs.py plot_curve work_dirs/soft_tgt.json work_dirs/default_stgcn/default.json \
# --keys top5_acc --legend soft_tgt default --iters-per-epoch 627 2506 --out top5_acc.pdf

