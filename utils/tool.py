import os
import time
import editdistance
from typing import List
from datetime import timedelta
import torch
import pandas as pd
import matplotlib.pyplot as plt

def current():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def elaspe(start: float):
    return timedelta(seconds=int(time.time() - start))

def error_rate(predicts_text: List[str], targets_text: List[str]) -> List[float]:
    pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predicts_text, targets_text)]
    er = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in pairs]
    return er

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total / 1e6))

def plot_curve(csv_path: str):
    df = pd.read_csv(csv_path)
    cer = df.loc[:, ['cer']].values
    train_loss = df.loc[:, ['train_loss']].values

    plt.plot(cer, label='cer', marker='*')
    plt.plot(train_loss, label='train_loss', marker='*')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.splitext(csv_path)[0] + '.png')
    
class Logger():
    def __init__(self, output_dir, model_name, mode='v2c') -> None:
        if mode == 'v2p':
            columns = ['epoch', 'lr', 'train_loss', 'per']
        elif mode == 'v2c':
            columns = ['epoch', 'lr', 'train_loss', 'cer']
        elif mode == 'v2pc':
            columns = ['epoch', 'lr', 'train_loss', 'per', 'cer']
        
        self.path = os.path.join(output_dir, model_name + '.csv')
        if os.path.exists(self.path):
            return
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.path, index=False)
    
    def append(self, *args):
        row = [list(args)]
        df = pd.DataFrame(row)
        df.to_csv(self.path, mode='a', index=False, header=False)
    
if __name__ == "__main__":
    log = Logger('./', 'c3d_seq', 'v2c')
    log.append(1, 0.001, 0.4, 0.3)
    log.append(2, 0.001, 0.3, 0.2)
    log.append(3, 0.001, 0.1, 0.1)
    
    plot_curve('c3d_seq.csv')