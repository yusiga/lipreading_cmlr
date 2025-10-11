import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PackedEncoder(nn.Module):
    def __init__(
        self,
        enc_input_size=512, 
        enc_hidden_size=256, 
        n_layers=2, 
        dropout=0.5
    ):
        super().__init__()
        self.gru = nn.GRU(enc_input_size, enc_hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self._init()

    def forward(self, input, length):
        """
        @param input: (T, B, 512)
        
        @return output_unpacked: (T, B, 2 * 256)
        @return hidden: (n_layer * 2, B, 256)
        """
        total_length = input.size(0)
        input_packed = pack_padded_sequence(input, length, enforce_sorted=False)

        self.gru.flatten_parameters()
        output_unpacked, hidden = self.gru(input_packed)
        output_unpacked, lens_unpacked = pad_packed_sequence(output_unpacked, total_length=total_length)

        return output_unpacked, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)

class Encoder(nn.Module):
    def __init__(
        self,
        enc_input_size=512, 
        enc_hidden_size=256, 
        n_layers=2, 
        dropout=0.5
    ):
        super().__init__()

        self.gru = nn.GRU(enc_input_size, enc_hidden_size, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self._init()

    def forward(self, input):
        """
        @param input: (T, B, 512)
        
        @return output: (T, B, 2 * 256)
        @return hidden: (n_layer * 2, B, 256)
        """        
        self.gru.flatten_parameters()
        output, hidden = self.gru(input)

        return output, hidden

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dec_input_size=512,
        dec_hidden_size=512,
        n_layers=2, 
        dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dec_input_size)
        self.gru = nn.GRU(dec_input_size, dec_hidden_size, n_layers, bidirectional=False)

        self.attention = Attention()
        self.attention_fc = nn.Linear(dec_hidden_size * 2, dec_hidden_size)
        self.fc = nn.Linear(dec_hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self._init()

    def forward(self, input, hidden, enc_output):
        """
        @param input:      (1, B)
        @param hidden:     (2, B, 512)
        @param enc_output: (T, B, 512)
        
        @return output:    (1, B, vocab_size)
        @return hidden:    (2, B, 512)
        @return alpha:     (B, 1, T)   
        """
        embedded = self.embedding(input)    # (1, B, 512)
        embedded = self.dropout(embedded)

        self.gru.flatten_parameters()
        # (1, B, 512), (2, B, 512)
        output, hidden = self.gru(embedded, hidden)  

        # (1, B, 512), (B, 1, T)
        context, alpha = self.attention.forward(hidden, enc_output)

        output = self.attention_fc(torch.cat((output, context), dim=2))
        output = self.relu(output)
        output = self.fc(output)    # (1, B, vocab_size)

        return output, hidden, alpha

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
    Reference: https://github.com/bentrevett/pytorch-seq2seq
    """
    def __init__(
        self,
        enc_hidden_size=256,
        dec_hidden_size=512
    ):
        super().__init__()
        self.fc1 = nn.Linear(enc_hidden_size * 4 + dec_hidden_size, dec_hidden_size)
        self.fc2 = nn.Linear(dec_hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self._init()

    def forward(self, hidden, enc_output):
        """
        @param hidden:      (2, B, 512)
        @param enc_output:  (T, B, 512)

        @return context:    (1, B, 512)
        @return alpha:      (B, 1, T)
        """
        T = enc_output.size(0)
        B = enc_output.size(1)

        hidden = hidden.permute(1, 0, 2).contiguous()           # (B, 2, 512)
        hidden = hidden.view(B, -1)                             # (B, 1024)
        hidden = hidden.repeat(T, 1, 1)                         # (T, B, 1024)
        hidden = hidden.permute(1, 0, 2).contiguous()           # (B, T, 1024)
        enc_output = enc_output.permute(1, 0, 2).contiguous()   # (B, T, 512)
        concat = torch.cat((hidden, enc_output), dim=2)         # (B, T, 1536) 
        
        alpha = self.tanh(self.fc1(concat))                     # (B, T, 512)
        alpha = self.fc2(alpha)                                 # (B, T, 1)
        alpha = alpha.permute(0, 2, 1).contiguous()             # (B, 1, T)
        alpha = self.softmax(alpha)
        
        context = torch.bmm(alpha, enc_output)                  # (B, 1, 512)
        context = context.permute(1, 0, 2).contiguous()         # (1, B, 512)
        
        return context, alpha

    def _init(self):
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)