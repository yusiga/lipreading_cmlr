import random
import torch
import torch.nn as nn

from ..attention import CA, SATA
from ..modules import PackedEncoder, Decoder


class Cnn3d(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.bn3 = nn.BatchNorm3d(96)

        self.sata = SATA()

        self.gru1 = nn.GRU(3072, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self._init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        """perform SATA"""
        x = self.sata.forward(x)  # (B, T, 3072)

        x = x.permute(1, 0, 2).contiguous()
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        return x

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class LipReader(nn.Module):
    def __init__(
            self,
            vocab_size,
            enc_input_size=512, enc_hidden_size=256,
            dec_input_size=512, dec_hidden_size=512,
            n_layers=2, dropout=0.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_hidden_size = enc_hidden_size
        self.n_layers = n_layers

        self.cnn = Cnn3d()
        self.encoder = PackedEncoder(enc_input_size, enc_hidden_size, n_layers, dropout)
        self.decoder = Decoder(vocab_size, dec_input_size, dec_hidden_size, n_layers, dropout)

    def forward(self, video, length, target):
        """
        @param video: (B, 3, T, H, W)
        @param length: T
        @param target: (L, B)
        """
        L, B = target.shape
        features = self.cnn.forward(video)  # (T, B, 512)

        # (T, B, 512), (n_layers * 2, B, 256)
        enc_output, enc_hidden = self.encoder.forward(features, length)

        dec_hidden = enc_hidden.permute(1, 0, 2).contiguous()  # (B, n_layers * 2, 256)
        dec_hidden = dec_hidden.view(-1, self.n_layers, self.enc_hidden_size * 2)  # (B, n_layers, 512)
        dec_hidden = dec_hidden.permute(1, 0, 2).contiguous()  # (n_layers, B, 512)

        dec_input = torch.ones(1, B, dtype=torch.long).cuda()
        output = torch.empty(L, B, self.vocab_size).cuda()
        predict = torch.empty(L, B).cuda()

        for i in range(L):
            dec_output, dec_hidden, alpha = self.decoder.forward(dec_input, dec_hidden, enc_output)
            output[i] = dec_output  # (1, B, vocab_size)

            top1 = torch.argmax(dec_output, dim=-1)
            predict[i] = top1  # (1, B)

            '''scheduled sampling: https://arxiv.org/pdf/1506.03099.pdf'''
            teacher_force = random.random() < 0.5
            dec_input = target[i].view(1, -1) if teacher_force else top1

        return output, predict

    def beam_search(self, video, length, k=1):
        pass

    def greedy_search(self, video, length, max_len):
        B = video.size(0)
        L = max_len
        features = self.cnn.forward(video)  # (T, B, 512)

        # (T, B, 512), (n_layers * 2, B, 256)
        enc_output, enc_hidden = self.encoder.forward(features, length)

        dec_hidden = enc_hidden.permute(1, 0, 2).contiguous()  # (B, n_layers * 2, 256)
        dec_hidden = dec_hidden.view(-1, self.n_layers, self.enc_hidden_size * 2)  # (B, n_layers, 512)
        dec_hidden = dec_hidden.permute(1, 0, 2).contiguous()  # (n_layers, B, 512)

        dec_input = torch.ones(1, B, dtype=torch.long).cuda()
        output = torch.empty(L, B, self.vocab_size).cuda()
        predict = torch.empty(L, B).cuda()

        for i in range(L):
            dec_output, dec_hidden, alpha = self.decoder.forward(dec_input, dec_hidden, enc_output)
            output[i] = dec_output  # (1, B, vocab_size)

            top1 = torch.argmax(dec_output, dim=-1)
            predict[i] = top1  # (1, B)

            dec_input = top1

        return output, predict
