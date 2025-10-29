import torch
import torch.nn as nn
from torch import Tensor
from .attention import CA, STA, SATA
from typing import Optional, Tuple


class Cnn3d(nn.Module):
    def __init__(self, dropout=0.1):
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

        self.ca = CA(96)
        self.sta = STA()
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

        # x = x.permute(2, 0, 1, 3, 4).contiguous()   # (T, B, 96, 4, 8)
        # x = x.view(x.size(0), x.size(1), -1)        # (T, B, 3072)
        x = self.ca(x)
        x = self.sta(x)
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


class MVTransformer_Encoder(nn.Module):
    def __init__(self, d_model, nhead=8, d_ff=2048, dropout=0.1, num_enc_layers=6, num_dec_layers=6):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=False, norm_first=True)
        # decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout, batch_first=False, norm_first=True)
        # 视频编码器
        # self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        # 文本编码器
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.fake_text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        # self.fake_text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        self._reset_parameters()

    def forward(self, x: Tensor, pseduo_x: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None
                ) -> tuple[Tensor, Tensor]:
        # mem1 = self.encoder1.forward(src1, src_mask, src_key_padding_mask)
        # mem2 = self.encoder2.forward(src2, src_mask, src_key_padding_mask)
        # mem = mem1 + mem2
        real_text_encoded = self.text_encoder.forward(x)
        fake_text_encoded = self.fake_text_encoder.forward(pseduo_x)
        # output = self.decoder.forward(x, mem2, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

        # fake_pred_text = self.fake_text_decoder.forward(pseduo_x, mem2,
        #                                         tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask)

        # return output
        return real_text_encoded, fake_text_encoded

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
