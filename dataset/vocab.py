from typing import List
from torch import Tensor
from typing import List
from torch import Tensor


# 词汇表类，文字或拼音等符号和整数索引之间互相转换
class Vocab():
    PATH_PINYIN_VOCAB = 'config/cmlr/pinyin_tone_2080.txt'  # 拼音词表
    PATH_CHAR_VOCAB = 'config/cmlr/char_vocab_list.txt'  # 汉字词表
    PATH_GRID_VOCAB = 'config/grid/grid_vocab_list.txt'  # GRID 数据集词表
    PATH_LRS2_VOCAB = 'config/lrs2/lrs2_all_word.txt'  # LRS2 数据集词表
    # PATH_LRS2_PRETRAIN_VOCAB = '/home/u2023110533/data/ZBC/zbc-lipreading-cmlr/config/lrs2/lrs2_pretrain_all_word.txt'
    PATH_LRS2_PRETRAIN_VOCAB = 'config/lrs2/lrs2_pretrain_200.txt'  # LRS2 预训练词表

    def __init__(self, type: str) -> None:
        self.PAD_TOKEN = 0  # 18000
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2

        # pad 填充，sos 起始，eos 结束
        self.word2Idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx2Word = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.size = 3
        # pad:0,sos:1,eos:2 0-3 1-4

        if type == 'p':
            path_vocab = Vocab.PATH_PINYIN_VOCAB
        elif type == 'c':
            path_vocab = Vocab.PATH_CHAR_VOCAB
        elif type == 'g':
            path_vocab = Vocab.PATH_GRID_VOCAB
        elif type == 's':
            path_vocab = Vocab.PATH_LRS2_VOCAB
        elif type == 'ps':
            path_vocab = Vocab.PATH_LRS2_PRETRAIN_VOCAB

        # self.word2Idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '我': 3, '要': 4, '吃': 5, '饭': 6}
        # self.idx2Word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '我', 4: '要', 5: '吃', 6: '饭'}
        with open(path_vocab, 'r') as f:
            for line in f.readlines():
                word = line.strip()
                if word not in self.word2Idx:
                    self.word2Idx[word] = self.size
                    self.idx2Word[self.size] = word
                    self.size += 1

    # 文本转索引
    def word2Index(self, sentence: List[str]) -> List[int]:
        """
        @param sentence
            >>> e.g. pinyin: ['wo', 'yao', 'chi', 'fan']
            >>> e.g. char: ['我', '要', '吃', '饭']
        """
        result = []
        for word in sentence:
            if word in self.word2Idx:
                result.append(self.word2Idx[word])
            else:
                result.append(self.PAD_TOKEN)
        return result

    # 索引转文本
    def index2Word(self, seq: Tensor) -> List[str]:
        """
        @param seq: [L, B]
        """
        result = []
        for i in range(seq.shape[1]):
            sentence = []
            for j in seq[:, i]:
                if j == self.EOS_TOKEN:
                    sentence.append('<eos>')
                    break
                elif j == self.PAD_TOKEN:
                    continue
                else:
                    sentence.append(self.idx2Word[j.item()])
            result.append(' '.join(sentence).strip())
        return result
