from typing import List
from torch import Tensor
from typing import List
from torch import Tensor


class Vocab():
    
    PATH_PINYIN_VOCAB = 'config/cmlr/pinyin_tone_2080.txt'
    PATH_CHAR_VOCAB = 'config/cmlr/char_vocab_list.txt'
    PATH_GRID_VOCAB = 'config/grid/grid_vocab_list.txt'
    PATH_LRS2_VOCAB = 'config/lrs2/lrs2_all_word.txt'
    # PATH_LRS2_PRETRAIN_VOCAB = '/home/u2023110533/data/ZBC/zbc-lipreading-cmlr/config/lrs2/lrs2_pretrain_all_word.txt'
    PATH_LRS2_PRETRAIN_VOCAB = 'config/lrs2/lrs2_pretrain_200.txt'
    def __init__(self, type: str) -> None:
        self.PAD_TOKEN = 0 #18000
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2
        
        self.word2Idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx2Word = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.size = 3
        #pad:0,sos:1,eos:2 0-3 1-4

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

        with open(path_vocab, 'r') as f:
            for line in f.readlines():
                word = line.strip()
                if word not in self.word2Idx:
                    self.word2Idx[word] = self.size
                    self.idx2Word[self.size] = word
                    self.size += 1
    
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