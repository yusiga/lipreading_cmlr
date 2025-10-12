import math
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet18
from ..attention import CA, STA, SATA
from typing import Optional
from ..vpd import Cnn3d, MVTransformer_Encoder
from ..tsd import MVTransformer_Decoder


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 500):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        # 防止位置编码过拟合
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# 频率分离，在时域上对视频帧进行 FFT & iFFT 变换
class FrequencySeparationModule(nn.Module):
    def __init__(self, channels, fps=30, freq_low=0.4, freq_high=3.0):
        super(FrequencySeparationModule, self).__init__()
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high

        self.conv_high = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, H, W, T]

        x = self.conv1(x)

        x_fft = torch.fft.fft(x, dim=-1)

        freqs = torch.fft.fftfreq(T, d=1.0 / self.fps).to(x.device)
        mask_high = ((torch.abs(freqs) >= self.freq_low) & (torch.abs(freqs) <= self.freq_high)).float()
        mask_low = 1.0 - mask_high

        mask_high = mask_high.view(1, 1, 1, 1, T)
        mask_low = mask_low.view(1, 1, 1, 1, T)

        high_fft = x_fft * mask_high
        low_fft = x_fft * mask_low

        high_fft = high_fft.permute(0, 4, 1, 2, 3)  # [B, T, C, H, W]
        low_fft = low_fft.permute(0, 4, 1, 2, 3)

        high = self.conv_high(high_fft.mean(dim=1).abs())  # [B, C, H, W]
        low = self.conv_low(low_fft.mean(dim=1).abs())

        high = torch.fft.ifft(high, dim=-1).real
        low = torch.fft.ifft(low, dim=-1).real

        return high, low  # [B, C, H, W]


# 唇部运动放大
class LipMotionMagnification(nn.Module):
    def __init__(self, channels, amp_factor=10):
        super(LipMotionMagnification, self).__init__()
        self.amp_factor = amp_factor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):  # [B, C, H, W]
        x = self.conv1(x)
        x = self.relu1(x)
        x = x * self.amp_factor  # 乘放大系数
        x = self.conv2(x)
        x = self.relu2(x)
        return x  # [B, C, H, W]


# 频率相关的分支
class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=64, out_channels=32, fps=30):
        super(FrequencyBranch, self).__init__()
        self.conv = nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.fsm = FrequencySeparationModule(channels=16, fps=fps)
        self.lmm = LipMotionMagnification(channels=16)
        # self.out_proj = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):  # [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.relu(self.conv(x))  # [B, 16, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, 16, H, W]
        high, low = self.fsm(x)  # [B, 16, H, W]
        x_lmm = self.lmm(low)  # [B, 16, H, W]
        x_out = x_lmm + low + high
        return x_out.view(x_out.size(0), -1)  # 展平 [B, N]


# 特征对齐层
class FeatureAlignment(nn.Module):
    def __init__(self, in1, in2, out_dim):
        super(FeatureAlignment, self).__init__()
        self.fc1 = nn.Linear(in1, out_dim)
        self.fc2 = nn.Linear(in2, out_dim)

    def forward(self, feat1, feat2):
        return self.fc1(feat1), self.fc2(feat2)


# 交叉注意力融合层
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)
        # matmul = matrix multiplication 矩阵乘法，transpose 维度交换，这里相当于矩阵倒置
        # attn_weights = self.softmax(Q @ K.T)
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5))
        # attn_output = attn_weights @ V
        attn_output = torch.matmul(attn_weights, V)
        # return x1 + attn_output
        output = self.output(attn_output) + x1
        return output


# 主模型
class LinProVSR(nn.Module):
    def __init__(self, n_pinyin, n_hanzi, d_model=512):
        super().__init__()
        self.max_target_len = 30
        self.d_model = d_model

        self.cnn3d = Cnn3d()
        # self.cnn2d = Cnn2d()
        self.pos_embed = PositionalEncoding(d_model)
        # 离散 → 连续
        self.embed = nn.Embedding(n_pinyin, d_model)

        self.branch_freq = FrequencyBranch()
        self.align_p = FeatureAlignment(in1=512, in2=6144, out_dim=512)
        self.fusion_tf_p = CrossAttentionFusion(dim=512)
        self.fusion_ft_p = CrossAttentionFusion(dim=512)
        self.align = FeatureAlignment(in1=512, in2=6144, out_dim=512)
        self.fusion_tf = CrossAttentionFusion(dim=512)
        self.fusion_ft = CrossAttentionFusion(dim=512)

        # self.transformer = MVTransformer(d_model)
        self.transformer_pinyin_En = MVTransformer_Encoder(d_model)
        self.transformer_pinyin_De = MVTransformer_Decoder(d_model)
        self.transformer_En = MVTransformer_Encoder(d_model)
        self.transformer_De = MVTransformer_Decoder(d_model)

        self.fc = nn.Linear(d_model, n_hanzi)
        self.fc_pinyin = nn.Linear(d_model, n_pinyin)

    def forward(self, video: Tensor, pinyin: Tensor, pseduo_pinyin: Tensor):
        """
        @video (B, 3, T, H, W)
        @pinyin (B, L)
        @return (L, B, vocab)
        """
        # 生成掩码
        # 在解码阶段屏蔽未来信息，预测第 t 个位置时，只能用前 0~t-1 的信息
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(pinyin.size(1)).cuda()  # (L, L)
        # 填充位置屏蔽，忽略为保证序列长度一致而填充的 padding
        tgt_pad_mask = (pinyin == 0).cuda()  # (B, L)

        # emb2d = self.pos_embed(self.cnn2d(video))
        # emb3d = self.pos_embed(self.cnn3d(video))
        video_c3d = video.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

        # 提取视频特征
        mem_pinyin = self.transformer_pinyin_En.encoder2(self.pos_embed(self.cnn3d(video_c3d)))
        mem_hanzi = self.transformer_En.encoder2(self.pos_embed(self.cnn3d(video_c3d)))

        # 提取频率特征，特征对齐 + 交叉融合
        feat_freq = self.branch_freq(video)
        # print(f'freq:{feat_freq.shape}') # [B,32]
        feat_time_p, feat_freq_p = self.align_p(mem_pinyin, feat_freq)
        fused_tf_p = self.fusion_tf_p(feat_time_p, feat_freq_p)
        fused_ft_p = self.fusion_ft_p(feat_freq_p, feat_time_p)
        fused_p = fused_tf_p + fused_ft_p

        feat_time, feat_freq = self.align(mem_hanzi, feat_freq)
        fused_tf = self.fusion_tf(feat_time, feat_freq)
        fused_ft = self.fusion_ft(feat_freq, feat_time)
        fused = fused_tf + fused_ft

        pinyin = pinyin.transpose(0, 1)  # (L, B)
        pseduo_pinyin = pseduo_pinyin.transpose(0, 1)  # (L, B)

        # 嵌入拼音序列
        pinyin = self.pos_embed(self.embed(pinyin))  # (L, B, d)
        pseduo_pinyin = self.pos_embed(self.embed(pseduo_pinyin))  # (L, B, d)

        # Transformer 编码 + 解码
        real_text_encoded, fake_text_encoded = self.transformer_pinyin_En.forward(
            x=pinyin,
            pseduo_x=pseduo_pinyin,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        output_pinyin, fake_pred_text = self.transformer_pinyin_De.forward(

            mem2=fused_p,
            x=pinyin,
            pseduo_x=pseduo_pinyin,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        # 输出线性映射到词表维度
        output_fc_pinyin = self.fc_pinyin(output_pinyin)
        output_fc_fake_pinyin = self.fc_pinyin(fake_pred_text)

        real_text_encoded_hanzi, fake_text_encoded_hanzi = self.transformer_En.forward(
            x=pinyin,
            pseduo_x=pseduo_pinyin,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        output_hanzi, fake_pred_text_hanzi = self.transformer_De.forward(
            mem2=fused,
            x=pinyin,
            pseduo_x=pseduo_pinyin,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        output_fc_hanzi = self.fc(output_hanzi)
        output_fc_fake = self.fc(fake_pred_text_hanzi)

        # output_hanzi, mem2_hanzi,real_text_encoded_hanzi,fake_text_encoded_hanzi,fake_pred_text_hanzi = self.transformer.forward(
        #     # src1=emb2d,
        #     src2=emb3d,
        #     x=pinyin,
        #     pseduo_x=pseduo_pinyin,
        #     tgt_mask=tgt_mask, 
        #     tgt_key_padding_mask=tgt_pad_mask
        # )

        # output_fc_hanzi = self.fc(output_hanzi) 
        # output_fc_fake = self.fc(fake_pred_text_hanzi)

        # return output 
        return fused_p, real_text_encoded, fake_text_encoded, output_fc_pinyin, fused, real_text_encoded_hanzi, fake_text_encoded_hanzi, output_fc_hanzi, output_fc_fake, output_fc_fake_pinyin
        # return output_fc, mem1, mem2,real_text_encoded,fake_text_encoded,mem,pinyin,output,fake_pred_text   # (L, B, vocab)

    # def predict_batch(self, video,beam_width=3):
    #     """
    #     @x: video (B, 3, T, H, W)
    #     return (L, B)
    #     """
    #     B = video.size(0)
    #     mem1 = self.transformer_pinyin.encoder1(self.pos_embed(self.cnn2d(video)))
    #     mem2 = self.transformer_pinyin.encoder2(self.pos_embed(self.cnn3d(video)))
    #     mem_pinyin = mem1 + mem2

    #     mem3 = self.transformer.encoder1(self.pos_embed(self.cnn2d(video)))
    #     mem4 = self.transformer.encoder2(self.pos_embed(self.cnn3d(video)))
    #     mem_hanzi = mem3 + mem4

    #     # Initialize beam search variables
    #     beams_pinyin = [torch.ones((1, B), dtype=torch.long).cuda()]
    #     beams_hanzi = [torch.ones((1, B), dtype=torch.long).cuda()]
    #     scores_pinyin= torch.zeros(B).cuda()
    #     scores_hanzi= torch.zeros(B).cuda()

    #     # predict_pinyin = torch.ones((1, B), dtype=torch.long).cuda()
    #     # predict_char = torch.ones((1, B), dtype=torch.long).cuda()
    #     for i in range(self.max_target_len-1):
    #         all_candidates_pinyin = []
    #         all_candidates_char = []
    #         for j in range(len(beams_pinyin)):
    #             predict_pinyin = beams_pinyin[j]
    #             predict_char = beams_hanzi[j]
    #             tgt_mask = nn.Transformer.generate_square_subsequent_mask(predict_pinyin.size(0)).cuda()  # (L, L)           
    #             output = self.pos_embed(self.embed(predict_pinyin))
    #             output_pinyin = self.transformer_pinyin.decoder.forward(output, mem_pinyin, tgt_mask=tgt_mask) 
    #             # output_char = self.transformer.decoder.forward(output_pinyin, mem_hanzi, tgt_mask=tgt_mask) 
    #             output_char = self.transformer.decoder.forward(output, mem_hanzi, tgt_mask=tgt_mask) 
    #             output_pinyin = output_pinyin.transpose(0, 1)     # (B, L, d)
    #             output_pinyin = self.fc_pinyin(output_pinyin[:, -1])     # (B, vocab)        
    #             # next_word_pinyin = torch.argmax(output_pinyin, dim=1)
    #             log_probs = torch.log_softmax(output_pinyin, dim=1)  # Use log softmax for numerical stability
    #             #汉字
    #             output_char = output_char.transpose(0, 1)     # (B, L, d)
    #             output_char = self.fc(output_char[:, -1])     # (B, vocab)   
    #             log_probs_char = torch.log_softmax(output_char, dim=1)

    #             # Expand each candidate
    #             topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=1)
    #             for k in range(beam_width):
    #                 candidate = torch.cat([predict_pinyin, topk_indices[:, k].view(1, -1)], dim=0)
    #                 candidate_score = scores_pinyin[j] + topk_log_probs[:, k]
    #                 all_candidates_pinyin.append((candidate_score, candidate))

    #             # Expand each candidate
    #             topk_log_probs_char, topk_indices_char = torch.topk(log_probs_char, beam_width, dim=1)
    #             for k in range(beam_width):
    #                 candidate_char = torch.cat([predict_char, topk_indices_char[:, k].view(1, -1)], dim=0)
    #                 candidate_score_char = scores_hanzi[j] + topk_log_probs_char[:, k]
    #                 all_candidates_char.append((candidate_score_char, candidate_char))

    #         # Select the best beam_width candidates
    #         ordered = sorted(all_candidates_pinyin, key=lambda tup: tup[0].sum().item(), reverse=True)
    #         beams_pinyin = [x[1] for x in ordered[:beam_width]]
    #         scores_pinyin = torch.stack([x[0] for x in ordered[:beam_width]])

    #         # Select the best beam_width candidates
    #         ordered = sorted(all_candidates_char, key=lambda tup: tup[0].sum().item(), reverse=True)
    #         beams_hanzi = [x[1] for x in ordered[:beam_width]]
    #         scores_hanzi = torch.stack([x[0] for x in ordered[:beam_width]])

    #             # next_word_pinyin = next_word_pinyin.view(1, -1)
    #             # predict_pinyin = torch.cat([predict_pinyin, next_word_pinyin], dim=0)

    #             # output_char = output_char.transpose(0, 1)     # (B, L, d)
    #             # output_char = self.fc(output_char[:, -1])     # (B, vocab)        
    #             # next_word_char = torch.argmax(output_char, dim=1)
    #             # next_word_char = next_word_char.view(1, -1)
    #             # predict_char = torch.cat([predict_char, next_word_char], dim=0)

    #     return beams_hanzi[0][1:, :]
    def predict_batch(self, video):
        """
        @x: video (B, 3, T, H, W)
        return (L, B)
        """
        video_c3d = video.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]
        B = video_c3d.size(0)
        # mem1 = self.transformer_pinyin.encoder1(self.pos_embed(self.cnn2d(video)))
        mem2 = self.transformer_pinyin_En.encoder2(self.pos_embed(self.cnn3d(video_c3d)))
        mem_pinyin = mem2

        # mem3 = self.transformer.encoder1(self.pos_embed(self.cnn2d(video)))
        mem4 = self.transformer_En.encoder2(self.pos_embed(self.cnn3d(video_c3d)))
        mem_hanzi = mem4

        feat_freq = self.branch_freq(video)

        feat_time_p, feat_freq_p = self.align_p(mem_pinyin, feat_freq)
        fused_tf_p = self.fusion_tf_p(feat_time_p, feat_freq_p)
        fused_ft_p = self.fusion_ft_p(feat_freq_p, feat_time_p)
        fused_p = fused_tf_p + fused_ft_p

        feat_time, feat_freq = self.align(mem_hanzi, feat_freq)
        fused_tf = self.fusion_tf(feat_time, feat_freq)
        fused_ft = self.fusion_ft(feat_freq, feat_time)
        fused = fused_tf + fused_ft

        predict_pinyin = torch.ones((1, B), dtype=torch.long).cuda()
        predict_char = torch.ones((1, B), dtype=torch.long).cuda()
        for i in range(self.max_target_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                predict_pinyin.size(0)).cuda()  # (L, L)
            output = self.pos_embed(self.embed(predict_pinyin))
            output_pinyin = self.transformer_pinyin_De.decoder.forward(output, fused_p, tgt_mask=tgt_mask)
            output_char = self.transformer_De.decoder.forward(output, fused, tgt_mask=tgt_mask)  # 不能注释pinyin,因为他会循环
            output_pinyin = output_pinyin.transpose(0, 1)  # (B, L, d)
            output_pinyin = self.fc_pinyin(output_pinyin[:, -1])  # (B, vocab)
            next_word_pinyin = torch.argmax(output_pinyin, dim=1)
            next_word_pinyin = next_word_pinyin.view(1, -1)
            predict_pinyin = torch.cat([predict_pinyin, next_word_pinyin], dim=0)

            output_char = output_char.transpose(0, 1)  # (B, L, d)
            output_char = self.fc(output_char[:, -1])  # (B, vocab)
            next_word_char = torch.argmax(output_char, dim=1)
            next_word_char = next_word_char.view(1, -1)
            predict_char = torch.cat([predict_char, next_word_char], dim=0)

        return predict_char[1:, :]
