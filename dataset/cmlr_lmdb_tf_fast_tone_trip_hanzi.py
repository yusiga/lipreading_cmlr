import pickle
import random
import numpy as np
import lmdb
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CMLR(Dataset):
    
    def __init__(self, path_list: str, env: object):
        self.env = env            
        with open(path_list) as f:
            self.item_list = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.item_list[index].encode()
            data = txn.get(key)
            
        data = pickle.loads(data)    
        video = data['video']   # numpy
        video = self.normalize(video, [0.4379, 0.4964, 0.6584], [0.1218, 0.1406, 0.1649])
        video = torch.from_numpy(video).float()
        # print(video)

        pinyin = data['pinyin']    # numpy
        pseudo_pinyin = data['pseudo_pinyin']
        pinyin = torch.from_numpy(pinyin).long()
        pseudo_pinyin = torch.from_numpy(pseudo_pinyin).long()
        char = data['char']    # numpy
        char = torch.from_numpy(char).long()
        # pinyin_input = torch.from_numpy(pinyin[:-1]).long()
        # pinyin_target = torch.from_numpy(pinyin[1:]).long()
        
        return video, pinyin,pseudo_pinyin,char
   
    def normalize(self, imgs, mean, std):
        imgs = imgs / 255.0
        imgs = (imgs - mean) / std
        return imgs

    @staticmethod
    def collate_fn(batch):
        videos, pinyin,pseudo_pinyin,char = zip(*batch)           # B * [T, H, W, C]
    
        videos = pad_sequence(videos, batch_first=True)         # [B, T, H, W, C]
        pinyin = pad_sequence(pinyin, batch_first=True)
        pseudo_pinyin = pad_sequence(pseudo_pinyin, batch_first=True)
        char = pad_sequence(char, batch_first=True)
        
        return videos,pinyin,pseudo_pinyin,char
