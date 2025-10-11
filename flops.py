import argparse
import time
import os
import math
import random
import numpy as np
import lmdb
import datetime
import importlib
from tqdm import tqdm
from loguru import logger
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from icecream import ic
import sys
from thop import profile,clever_format
from dataset.cmlr_lmdb_tf_fast_tone_trip_hanzi_cold import CMLR
from model.tf.c3d_c2d_tf_add_tone_pseduo_hanzi_pinyin_c3d import LipNet
from config.cmlr.config import args
from utils.util import Util
from dataset.vocab import Vocab

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Hyperparameters
batch_size = 16

# Set device to CPU
args.device = torch.device("cpu")

# Open LMDB environment
env = lmdb.open(args.path_cmlr_lmdb, readonly=True, lock=False)

# Load test dataset and create DataLoader
test_set = CMLR(args.path_valid_csv, env, args.path_feature)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, collate_fn=CMLR.collate_fn
)

# Load vocabularies and model
pinyin_vocab = Vocab('p')
char_vocab = Vocab('c')
model = LipNet(pinyin_vocab.size, char_vocab.size).to(args.device)

model = Util.load_model_state(
    model,
    "/home/u2023110533/data/ZBC/zbc-save-cmlr/20241203_111321-c3d_c2d_tf_add_tone_pseduo_hanzi_pinyin_c3d/best.pt",
)
def cal_flops_params(model,inputs):
    flops,params = profile(model,inputs=inputs)
    flops,params = clever_format([flops,params],"%.2f")
    return flops,params

for i, (video,pinyin,pseduo_pinyin,char,length) in enumerate(test_loader):
        video = video.to(args.device, non_blocking=True)
        pinyin_in = pinyin[:, :-1].to(args.device, non_blocking=True)     # [B, L]
        pseduo_pinyin = pseduo_pinyin[:, :-1].to(args.device, non_blocking=True)     # [B, L]
        char_tgt = char[:, 1:].to(args.device, non_blocking=True)   # [B, L]
        pinyin_tgt = pinyin[:, 1:].to(args.device, non_blocking=True)   # [B, L]
        total_flops,total_params = cal_flops_params(model,inputs=(video,pinyin_in,pseduo_pinyin))
        print(total_flops.total_params)
# if __name__ == "__main__":
#     total_flops,total_params = cal_flops_params(model,inputs=(video,pinyin_in,pseduo_pinyin))