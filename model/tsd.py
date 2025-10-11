import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class MVTransformer_Decoder(nn.Module):
    def __init__(self, d_model, nhead=8, d_ff=2048, dropout=0.1, num_enc_layers=6, num_dec_layers=6):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout, batch_first=False, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        self.fake_text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        
        self._reset_parameters()

    def forward(self,mem2: Tensor, x: Tensor, pseduo_x:Tensor,
                src_mask: Optional[Tensor] = None, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        output = self.decoder.forward(x, mem2, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
      
        fake_pred_text = self.fake_text_decoder.forward(pseduo_x, mem2,
                                                tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask)
        
        # return output
        return output,fake_pred_text

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
