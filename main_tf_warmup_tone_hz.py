import argparse
import time
import os
import math
import random
import numpy as np
import lmdb
import csv

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# data
from dataset.vocab import Vocab
from dataset.cmlr_lmdb_tf_fast_tone_trip_hanzi import CMLR

# model
from model.tf.tf_add_tone_pseduo_hz import LinProVSR
import itertools

# tool
from utils.tool import *

from utils.loss import triplet_loss

NUM_EPOCH = 200
WARMUP_EPOCH = 5
LERANING_RATE = 3e-4
# 6e-3
BATCH_SIZE = 16
BATCH_SIZE_VALID = 8

PATH_TRAIN_LIST = 'config/cmlr/train_list.txt'
PATH_VALID_LIST = 'config/cmlr/valid_list.txt'
PATH_TEST_LIST = 'config/cmlr/test_list.txt'

PREFIX = '/data'
SAVE_PREFIX = '/home/hfut/ZJR'
# PATH_LMDB = f'{PREFIX}/Dataset/CMLR_lmdb'
PATH_LMDB = f'{PREFIX}/Dataset/CMLR_lmdb_tone_pseduo'
PATH_SAVE = f'{SAVE_PREFIX}/ZBC/zbc-save-cmlr'

parser = argparse.ArgumentParser(description='Lipreading Train')
# 是否使用预训练权重
parser.add_argument('--use-ckpt', type=int, default=0, help='if use pretrained checkpoint to train')
# 预训练模型的路径
parser.add_argument('--path-ckpt', type=str, default='', help='path of pretrained checkpoint')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
为了确保实验的可重复性，对可能用到的随机数，设置随机种子
"""
torch.manual_seed(0)  # 设置CPU生成随机数的种子
torch.cuda.manual_seed(0)  # 设置GPU生成随机数的种子
torch.cuda.manual_seed_all(0)  # 设置所有GPU生成随机数的种子，如果使用多卡
np.random.seed(0)  # 设置numpy生成随机数的种子
random.seed(0)  # 设置random生成随机数的种子

pinyin_vocab = Vocab('p')  # 创建一个实例对象
char_vocab = Vocab('c')

print(f"pinyin_vocab Length: {pinyin_vocab.size}")
print(f"char_vocab Length: {char_vocab.size}")

print("vocabulary loaded successfully")
"""
LMDB是一个键值存储数据库,常用于机器学习项目中存储大量数据，
因为它支持快速的读写操作，并且可以通过内存映射进行高效的数据访问
LMDB要求键和值都是字节类型
"""
env = lmdb.open(PATH_LMDB, readonly=True, lock=False)
train_set = CMLR(PATH_TRAIN_LIST, env)
valid_set = CMLR(PATH_VALID_LIST, env)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8,
                          collate_fn=CMLR.collate_fn)
valid_loader = DataLoader(valid_set, BATCH_SIZE_VALID, shuffle=False, pin_memory=True, num_workers=8,
                          collate_fn=CMLR.collate_fn)

print(f"train set: {len(train_set)}")
print(f"valid set: {len(valid_set)}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinProVSR(pinyin_vocab.size, char_vocab.size).to(DEVICE)
name = str(type(model)).split('.')[-2]

criteron = nn.CrossEntropyLoss(ignore_index=0)  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=LERANING_RATE, amsgrad=True)  # 优化器

# 学习率相关
warmup = lambda epoch: (epoch + 1) / WARMUP_EPOCH if epoch < WARMUP_EPOCH else 1.0
scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
scheduler_reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=1)

if args.use_ckpt == 1:
    # using previous checkpoint
    ckpt = torch.load(args.path_ckpt)
    start_epoch = ckpt['epoch'] + 1
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler_reduce.load_state_dict(ckpt['lr_scheduler'])
    del ckpt
    # using previous output directory
    output_dir = os.path.dirname(args.path_ckpt)
    print('\033[1;32m', output_dir, '\033[0m')
    print('\033[1;36m', 'train will resume from checkpoint', '\033[0m')
else:
    start_epoch = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = f'{PATH_SAVE}/{timestamp}-{name}'
    os.makedirs(output_dir, exist_ok=True)

    print('\033[1;32m', output_dir, '\033[0m')
    print('\033[1;36m', 'train will start from scratch', '\033[0m')

log = Logger(output_dir, name)


def train_epoch(epoch):
    loss_sum = 0
    model.train()
    train_results = []
    # 所有可能的权重组合
    # w_pinyin_options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # w_char_options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # w_trip_options = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.5]
    # w_trip_out_options = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.5]
    # w_pinyin_trip_options = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.5]
    # w_pinyin_trip_out_options = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.5]

    # best_loss = float('inf')
    # best_weights = None
    for i, (video, pinyin, pseduo_pinyin, char) in enumerate(train_loader):
        video = video.to(DEVICE, non_blocking=True)
        pinyin_in = pinyin[:, :-1].to(DEVICE, non_blocking=True)  # [B, L]
        pseduo_pinyin = pseduo_pinyin[:, :-1].to(DEVICE, non_blocking=True)  # [B, L]
        char_tgt = char[:, 1:].to(DEVICE, non_blocking=True)  # [B, L]
        pinyin_tgt = pinyin[:, 1:].to(DEVICE, non_blocking=True)  # [B, L]

        optimizer.zero_grad()  # 梯度清零
        mem2, real_text_encoded, fake_text_encoded, output_fc_pinyin, mem2_hanzi, real_text_encoded_hanzi, fake_text_encoded_hanzi, output_fc_hanzi, output_fc_fake, output_fc_fake_pinyin = model.forward(
            video, pinyin_in, pseduo_pinyin)  # (L, B, vocab)
        output_fc_pinyin = output_fc_pinyin.permute(1, 2, 0)  # (B, vocab, L)
        output_fc_fake_pinyin = output_fc_fake_pinyin.permute(1, 2, 0)
        output_fc_hanzi = output_fc_hanzi.permute(1, 2, 0)  # (B, vocab, L)
        output_fc_fake = output_fc_fake.permute(1, 2, 0)
        """
        output_hanzi:真实文本和真实视频解码出的结果
        mem1_hanzi:2D视频特征编码后
        mem2_hanzi:3D视频特征编码后
        mem_hanzi:融合后的特征
        real_text_encoded_hanzi:真实文本标签经过文本编码器编码后的
        fake_text_encoded: 伪文本标签经过文本编码器编码后的
        fake_pred_text_hanzi:伪文本和真实视频解码出的结果
        """

        # for w_pinyin, w_char, w_trip, w_trip_out, w_pinyin_trip, w_pinyin_trip_out in itertools.product(w_pinyin_options, w_char_options, w_trip_options, w_trip_out_options, w_pinyin_trip_options, w_pinyin_trip_out_options):
        # 初始化权重
        # w_pinyin = torch.tensor(w_pinyin, requires_grad=False)
        # w_char = torch.tensor(w_char, requires_grad=False)
        # w_trip = torch.tensor(w_trip, requires_grad=False)
        # w_trip_out = torch.tensor(w_trip_out, requires_grad=False)
        # w_pinyin_trip = torch.tensor(w_pinyin_trip, requires_grad=False)
        # w_pinyin_trip_out = torch.tensor(w_pinyin_trip_out, requires_grad=False)

        loss_vpc = triplet_loss(mem2.transpose(0, 1).mean(dim=1), real_text_encoded.transpose(0, 1).mean(dim=1),
                                fake_text_encoded.transpose(0, 1).mean(dim=1))  # 视频编码，真文本编码，假文本编码（拼音版）
        # loss_tsc = triplet_loss(pinyin_tgt,output_fc_pinyin.mean(dim=1),output_fc_fake_pinyin.mean(dim=1)) # 标签，真预测，假预测

        # loss_trip = triplet_loss(mem2_hanzi.transpose(0,1).mean(dim=1),real_text_encoded_hanzi.transpose(0,1).mean(dim=1),fake_text_encoded_hanzi.transpose(0,1).mean(dim=1))# 视频编码，真文本编码，假文本编码
        # loss_trip_out = triplet_loss(char_tgt.mean(dim=1),output_fc_hanzi.mean(dim=1),output_fc_fake.mean(dim=1)) # 标签，真预测，假预测
        # print(char_tgt.shape,output_fc_hanzi.mean(dim=1).shape,output_fc_fake.mean(dim=1).shape)
        # loss_trip_out = triplet_loss(char_tgt,output_fc_hanzi.mean(dim=1),output_fc_fake.mean(dim=1)) # 标签，真预测，假预测

        loss_pinyin_ce = criteron(output_fc_pinyin, pinyin_tgt)
        loss_char_ce = criteron(output_fc_hanzi, char_tgt)
        # 根据权重计算总损失
        loss = (
                loss_char_ce +
                loss_vpc +
                loss_pinyin_ce
            # loss_tsc
        )

        # # 如果当前损失更小，保存当前权重
        # if loss < best_loss:
        #     best_loss = loss
        #     best_weights = (w_pinyin.item(), w_char.item(), w_trip.item(), w_trip_out.item(), w_pinyin_trip.item(), w_pinyin_trip_out.item())

        # loss = loss_pinyin_ce + loss_char_ce+0.1*loss_trip+0.1*loss_trip_out+0.1*loss_pinyin_ce_trip+0.1*loss_pinyin_ce_trip_out
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新参数

        loss = loss.item()
        loss_sum += loss

        # Convert output to predicted text
        predict = output_fc_hanzi.argmax(dim=1)
        # predict = output_hanzi.argmax(dim=1)  
        predict_text = char_vocab.index2Word(predict)
        target_text = char_vocab.index2Word(char_tgt)

        # Append predictions and targets to results list
        for pt, tt in zip(predict_text, target_text):
            train_results.append([pt, tt])

        if i % 100 == 0:
            print(f'epoch-{epoch} [{i * BATCH_SIZE:>5d}/{len(train_set)}], lr={lr:.8f}, loss={loss:.6f}, {current()}')
            # print(f'Best weights: w_pinyin={best_weights[0]}, w_char={best_weights[1]}, w_trip={best_weights[2]}, w_trip_out={best_weights[3]}, w_pinyin_trip={best_weights[4]}, w_pinyin_trip_out={best_weights[5]}')

    # Save results to CSV file
    # csv_filename = f'{CSV_OUTPUT_PATH}/train_predictions_vs_targets_epoch_{epoch}.csv'
    # with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Prediction', 'Target'])
    #     writer.writerows(train_results)
    return loss_sum / len(train_loader)


@torch.no_grad()
def valid_epoch():
    cer_sum = 0
    # torch.cuda.empty_cache()
    model.eval()
    valid_results = []
    for i, (video, pinyin, pseduo_pinyin, char) in enumerate(valid_loader):
        video = video.to(DEVICE, non_blocking=True)
        char_tgt = char[:, 1:].to(DEVICE, non_blocking=True)  # [B, L]

        predict = model.predict_batch(video)  # [L, B]
        predict_text = char_vocab.index2Word(predict)
        target_text = char_vocab.index2Word(char_tgt.transpose(0, 1))
        cer = error_rate(predict_text, target_text)
        cer_sum += sum(cer)
        for pt, tt in zip(predict_text, target_text):
            valid_results.append([pt, tt])

        if i % 100 == 0:
            print(f'P vs T [{i * BATCH_SIZE_VALID:>5d}/{len(valid_set)}]: {current()}')
            for predict, truth in zip(predict_text[:4], target_text[:4]):
                print('\033[1;35m', f'P: {predict:<70}', '\033[0m')
                print('\033[1;32m', f'T: {truth:<70}', '\033[0m')
                print('-' * 100)
    # Save results to CSV file
    # valid_csv_filename = f'{CSV_OUTPUT_PATH}/valid_predictions_vs_targets.csv'
    # with open(valid_csv_filename, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Prediction', 'Target'])
    #     writer.writerows(valid_results)

    return cer_sum / len(valid_set)


if __name__ == '__main__':
    start = time.time()
    cer = 1.00
    for epoch in range(start_epoch, NUM_EPOCH):
        lr = optimizer.param_groups[0]['lr']

        # train epoch
        print(f'epoch {epoch} train starts: {current()}')
        start_train = time.time()
        train_loss = train_epoch(epoch)
        # train_loss = 0
        print('-' * 80)
        print(f'epoch {epoch} train ends: {current()}, elaspe={elaspe(start_train)}')
        print('\033[1;33m', f'loss={train_loss:.6f}, lr={lr:.8f}', '\033[0m')
        print('-' * 80)

        # valid epoch
        print(f'epoch {epoch} valid starts: {current()}')
        start_valid = time.time()
        valid_cer = valid_epoch()
        print(f'epoch {epoch} valid ends: {current()}, elaspe={elaspe(start_valid)}')
        print('\033[1;33m', f'cer={valid_cer:.6f}', '\033[0m', f'loss={train_loss:.6f}, lr={lr:.8f}')
        print('-' * 80)

        # log record
        log.append(epoch, lr, train_loss, valid_cer)

        # update learning rate
        if epoch < WARMUP_EPOCH:
            scheduler_warmup.step()
        else:
            scheduler_reduce.step(valid_cer)

        # save checkpoint
        if cer > valid_cer:
            # save_path = os.path.join(output_dir, f'ep{epoch:0>2}_lr_{lr:.8f}_loss_{train_loss:.6f}_cer_{valid_cer:.6f}.pt')
            save_path = os.path.join(output_dir, f'best.pt')
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler_reduce.state_dict()
            }
            torch.save(ckpt, save_path)
            print(f'epoch {epoch} checkpoint saved, elaspe={elaspe(start_train)}, total={elaspe(start)}')
            print('-' * 80)
            cer = valid_cer

    env.close()
