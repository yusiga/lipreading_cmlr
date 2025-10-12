import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..attention import CA, SATA, STA


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

        # self.ca = CA(96)
        # self.sta = STA()
        self.sata = SATA()

        self.gru1 = nn.GRU(3072, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self._init()

    def forward(self, x: Tensor):
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

        """perform CA, STA, SATA"""
        x = self.sata(x)  # (B, T, 3072)

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


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(feature_size, hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self._init()

    def forward(self, input):
        # input: (T, B, H)

        self.gru.flatten_parameters()
        output, hidden = self.gru(input)
        # output: (T, B, H)
        # hidden: (n_layers * 2, B, H)
        return output, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)


class PackedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.5):
        super(PackedEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self._init()

    def forward(self, input, length):
        # input: (T, B, H)
        total_length = input.size()[0]
        input_packed = pack_padded_sequence(input, length, enforce_sorted=False)

        self.gru.flatten_parameters()
        output, hidden = self.gru(input_packed)

        output_unpacked, lens_unpacked = pad_packed_sequence(output, total_length=total_length)
        # output_unpacked: (T, B, H)
        # hidden: (n_layers * 2, B, H)
        return output_unpacked, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=False)

        self.attention = Attention(int(hidden_size / 2), n_layers=n_layers)
        self.attention_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self._init()

    def forward(self, input, hidden, encoder_output=None):
        # input: (1, B)
        # embedded: (1, B, hidden)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # output: (1, B, hidden_size)
        # hidden: (2, B, hidden_size)
        self.gru.flatten_parameters()
        output, hidden = self.gru(embedded, hidden)

        # Attention here.
        if encoder_output is not None:
            context, alpha = self.attention(hidden, encoder_output)
            output = self.attention_fc(torch.cat((output, context), dim=2))
            output = self.relu(output)

        # output: (1, B, output_size)
        output = self.fc(output)

        if encoder_output is not None:
            return output, hidden, alpha
        return output, hidden

    def _init(self):
        nn.init.kaiming_normal_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)


class Attention(nn.Module):
    """
    Reference: https://jaketae.github.io/study/seq2seq-attention/
    """

    def __init__(self, hidden_size, n_layers=2):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2 * (n_layers + 1), hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

        self._init()

    def forward(self, state, encoder_output):
        # encoder_output: (T, B, hidden_size * 2)
        # state: (n_layers * 2, B, hidden_size)
        seq_size = encoder_output.size()[0]
        batch_size = encoder_output.size()[1]

        # state: (B, hidden_size * n_layers * 2)
        state = state.permute(1, 0, 2).contiguous()
        state = state.view(batch_size, -1).contiguous()
        # state: (B, T, hidden_size * n_layers * 2)
        state = state.repeat(seq_size, 1, 1).permute(1, 0, 2).contiguous()
        # encoder_output: (B, T, hidden_size * 2)
        encoder_output = encoder_output.permute(1, 0, 2).contiguous()
        # concat: (B, T, hidden_size * 2 * (n_layers + 1))
        concat = torch.cat((state, encoder_output), dim=2)
        # energy: (B, T, hidden_size * 2)
        energy = self.tanh(self.fc1(concat))
        # alpha: (B, T, 1)
        energy = self.fc2(energy).permute(0, 2, 1).contiguous()
        # alpha: (B, 1, T)
        alpha = self.softmax(energy)
        # context: (1, B, hidden_size * 2)
        context = torch.bmm(alpha, encoder_output).permute(1, 0, 2).contiguous()
        return context, alpha
        # return context, energy

    def _init(self):
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)


class CascadeDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=2, dropout=0.5):
        super(CascadeDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.attention = Attention(int(hidden_size / 2), n_layers=n_layers)
        self.attention_p = Attention(int(hidden_size / 2), n_layers=n_layers)
        self.attention_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self._init()

    def forward(self, input, hidden, encoder_output=None, encoder_output_pinyin=None):
        # input: (1, B)
        # embedded: (1, B, hidden)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # output: (1, B, hidden_size)
        # hidden: (2, B, hidden_size)
        self.gru.flatten_parameters()
        output, hidden = self.gru(embedded, hidden)

        # Attention here.
        if encoder_output is not None and encoder_output_pinyin is not None:
            context, alpha = self.attention(hidden, encoder_output)
            context_p, alpha_p = self.attention_p(hidden, encoder_output_pinyin)
            output = self.attention_fc(torch.cat((output, context, context_p), dim=2))
            output = self.relu(output)

        # output: (1, B, output_size)
        output = self.fc(output)

        return output, hidden

    def _init(self):
        nn.init.kaiming_normal_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)


class LipReader(nn.Module):
    def __init__(
            self,
            feature_size,
            n_pinyin,
            n_hanzi,
            hidden_size,
            sos_token=1, n_layers=2, dropout=0.5
    ):
        super(LipReader, self).__init__()
        self.feature_size = feature_size
        self.n_pinyin = n_pinyin
        self.n_hanzi = n_hanzi
        self.hidden_size = hidden_size

        self.sos_token = sos_token
        self.n_layers = n_layers

        self.cnn = Cnn3d()

        # video to pinyin
        self.embedding = nn.Embedding(n_pinyin, feature_size)
        self.encoder_pinyin = PackedEncoder(feature_size, hidden_size, n_layers=n_layers, dropout=dropout)
        self.decoder_pinyin = Decoder(n_pinyin, hidden_size * 2, n_layers=n_layers, dropout=dropout)

        # video, pinyin to hanzi
        self.encoder_hanzi = Encoder(feature_size, hidden_size, n_layers=n_layers, dropout=dropout)
        self.decoder_hanzi = CascadeDecoder(n_hanzi, hidden_size * 2, n_layers=n_layers, dropout=dropout)

    def forward(self, input, length, target_pinyin, target_hanzi, teacher_forcing_ratio=0.5):
        """
        @input: (B, 3, T, H, W)
        """
        L, B = target_hanzi.shape

        # features: (T, B, 512)
        features = self.cnn(input)

        # encoder_output: (T, B, H)
        # encoder_hidden: (n_layers * 2, B, H)
        # decoder_hidden: (n_layers, B, H * 2)
        encoder_output, encoder_hidden = self.encoder_pinyin(features, length)

        decoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden = decoder_hidden.view(-1, self.n_layers, self.hidden_size * 2)
        decoder_hidden = decoder_hidden.permute(1, 0, 2).contiguous()

        # decoder_input: (1, B)
        # output_pinyin: (L, B, n_pinyin)
        # predict_pinyin: (L, B)
        decoder_input = torch.ones(1, B, dtype=torch.long).cuda()
        output_pinyin = torch.empty(L, B, self.n_pinyin).cuda()
        predict_pinyin = torch.empty(L, B).cuda()

        for i in range(L):
            # output_pinyin: (1, B, n_pinyin)
            # hidden: (2, B, hidden_size)
            decoder_output, decoder_hidden, alpha = self.decoder_pinyin(decoder_input, decoder_hidden,
                                                                        encoder_output)  # use attention
            output_pinyin[i] = decoder_output

            top1 = decoder_output.argmax(-1)
            predict_pinyin[i] = top1

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_pinyin[i].view(1, -1) if teacher_force else top1

        encoder_output_pinyin = encoder_output

        '''pinyin to hanzi'''
        # predict_pinyin: (T, B)
        # embedded: (T, B, F)
        predict_pinyin = predict_pinyin.long()
        embedded = self.embedding(predict_pinyin)

        # encoder_output: (T, B, H)
        # encoder_hidden: (n_layers * 2, B, H)
        # decoder_hidden: (n_layers, B, H * 2)
        encoder_output, encoder_hidden = self.encoder_hanzi(embedded)

        decoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden = decoder_hidden.view(-1, self.n_layers, self.hidden_size * 2)
        decoder_hidden = decoder_hidden.permute(1, 0, 2).contiguous()

        # decoder_input: (1, B)
        # output_hanzi: (L, B, n_hanzi)
        # predict_hanzi: (L, B)
        decoder_input = torch.ones(1, B, dtype=torch.long).cuda()
        output_hanzi = torch.empty(L, B, self.n_hanzi).cuda()
        predict_hanzi = torch.empty(L, B).cuda()

        for i in range(L):
            # output_hanzi: (1, B, n_hanzi)
            # hidden: (2, B, hidden_size)
            decoder_output, decoder_hidden = self.decoder_hanzi(decoder_input, decoder_hidden, encoder_output,
                                                                encoder_output_pinyin)  # use attention
            output_hanzi[i] = decoder_output

            top1 = decoder_output.argmax(-1)
            predict_hanzi[i] = top1

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_hanzi[i].view(1, -1) if teacher_force else top1

        return output_pinyin, predict_pinyin, output_hanzi, predict_hanzi

    def greedy_search(self, input, length, max_len):
        # input: (B, 3, T, H, W)
        B = input.size(0)
        L = max_len

        # feature_output: (T, B, 512)
        feature_output = self.cnn(input)

        # encoder_output: (T, B, H)
        # encoder_hidden: (n_layers * 2, B, H)
        # decoder_hidden: (n_layers, B, H * 2)
        encoder_output, encoder_hidden = self.encoder_pinyin(feature_output, length)
        decoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden = decoder_hidden.view(-1, self.n_layers, self.hidden_size * 2)
        decoder_hidden = decoder_hidden.permute(1, 0, 2).contiguous()

        # decoder_input: (1, B)
        # output_pinyin: (L, B, n_pinyin)
        # predict_pinyin: (L, B)
        decoder_input = torch.ones(1, B, dtype=torch.long).cuda()
        output_pinyin = torch.empty(L, B, self.n_pinyin).cuda()
        predict_pinyin = torch.empty(L, B).cuda()

        for i in range(L):
            # output_pinyin: (1, B, n_pinyin)
            # hidden: (2, B, hidden_size)
            decoder_output, decoder_hidden, alpha = self.decoder_pinyin(decoder_input, decoder_hidden,
                                                                        encoder_output)  # use attention
            output_pinyin[i] = decoder_output

            top1 = decoder_output.argmax(-1)
            predict_pinyin[i] = top1

            decoder_input = top1

        encoder_output_pinyin = encoder_output

        '''pinyin to hanzi'''
        # predict_pinyin: (T, B)
        # embedded: (T, B, F)
        predict_pinyin = predict_pinyin.long()
        embedded = self.embedding(predict_pinyin)

        # encoder_output: (T, B, H)
        # encoder_hidden: (n_layers * 2, B, H)
        # decoder_hidden: (n_layers, B, H * 2)
        encoder_output, encoder_hidden = self.encoder_hanzi(embedded)

        decoder_hidden = encoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden = decoder_hidden.view(-1, self.n_layers, self.hidden_size * 2)
        decoder_hidden = decoder_hidden.permute(1, 0, 2).contiguous()

        # decoder_input: (1, B)
        # output_hanzi: (L, B, n_hanzi)
        # predict_hanzi: (L, B)
        decoder_input = torch.ones(1, B, dtype=torch.long).cuda()
        output_hanzi = torch.empty(L, B, self.n_hanzi).cuda()
        predict_hanzi = torch.empty(L, B).cuda()

        for i in range(L):
            # output_hanzi: (1, B, n_hanzi)
            # hidden: (2, B, hidden_size)
            decoder_output, decoder_hidden = self.decoder_hanzi(decoder_input, decoder_hidden, encoder_output,
                                                                encoder_output_pinyin)  # use attention
            output_hanzi[i] = decoder_output

            top1 = decoder_output.argmax(-1)
            predict_hanzi[i] = top1

            decoder_input = top1

        return output_pinyin, predict_pinyin, output_hanzi, predict_hanzi
