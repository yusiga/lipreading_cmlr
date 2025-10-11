#!/bin/bash
log_name=$(date "+%Y.%m.%d_%H.%M.%S")
# path_ckpt='/data/xyc-save-cmlr/20240229_173128-c3d_c2d_tf_add/ep10_lr_0.00030000_loss_0.346924_cer_0.329926.pt'

nohup python -u main_tf_warmup_tone_hz.py --use-ckpt=0 > log/${log_name}-1011-notsc.log &
# nohup python -u main_tf_warmup.py --use-ckpt=1 --path-ckpt=${path_ckpt} >> 2024.02.29_17.31.20.log &