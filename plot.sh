conda activate open-mmlab
python tools/analysis/analyze_logs.py plot_curve work_dirs/soft_tgt.json work_dirs/default_stgcn/default.json \
--keys top1_acc --legend soft_tgt default --iters-per-epoch 627 2506 --out top1_acc.pdf

