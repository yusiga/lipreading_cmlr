import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.modules.activation import MultiheadAttention

class CA(nn.Module):
    """
    Reference: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    def __init__(self, in_channel=96):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channel // 16, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        @param x: (B, C, T, H, W)
        @return out: (B, C, T, H, W)
        """
        avg_out = self.fc(self.avg_pool(x))     # (B, C, 1, 1, 1)
        max_out = self.fc(self.max_pool(x))     # (B, C, 1, 1, 1)
        out = self.sigmoid(avg_out + max_out) 
        out = out * x
        return out
    
class STA(nn.Module):
    """
    Reference: https://github.com/V-Sense/ACTION-Net/blob/main/models/action.py
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param x: (B, C, T, H, W)
        @return out: (B, C, T, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, T, H, W)
        out = self.conv(out)        # (B, 1, T, H, W)
        out = self.sigmoid(out)
        out = out * x
        return out
    
class SATA1(nn.Module):
    """
    separate Spatial Temporal Attention
    """
    def __init__(self, size=3):
        super().__init__()
        self.sp_conv = nn.Conv2d(96, 1, kernel_size=size, padding=size // 2)
        self.tp_conv = nn.Conv1d(3072, 1, kernel_size=size, padding=size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param (B, 96, T, 4, 8)
        @return (B, T, 3072)
        """
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, T, 96, 4, 8)
        
        '''spatial attention'''
        sp_x = x.view(-1, C, H, W)                      # (B*T, 96, 4, 8)

        sp_attention = self.sp_conv(sp_x)               # (B*T, 3, 4, 8)
        sp_attention1 = sp_attention[:, 0].unsqueeze(1)         
        sp_attention1 = self.sigmoid(sp_attention1)     # (B*T, 1, 4, 8)      
        sp_attention1 = sp_attention1 * sp_x            # (B*T, 96, 4, 8)
        sp_attention1 = sp_attention1.view(B, T, -1)    # (B, T, 3072)

        '''temporal attention'''
        tp_x = x.view(B, T, -1)                         # (B, T, 3072)
        tp_x = tp_x.permute(0, 2, 1).contiguous()       # (B, 3072, T)
        
        tp_attention = self.tp_conv(tp_x)               # (B, 3, T)
        tp_attention1 = tp_attention[:, 0].unsqueeze(1) 
        tp_attention1 = self.sigmoid(tp_attention1)     # (B, 1, T)
        tp_attention1 = tp_attention1 * tp_x            # (B, 3072, T)
        tp_attention1 = tp_attention1.permute(0, 2, 1).contiguous()
        
        '''add norm'''
        feature1 = F.normalize(sp_attention1 + tp_attention1, p=2, dim=2)

        return feature1

class SATA2(nn.Module):
    """
    separate Spatial Temporal Attention
    """
    def __init__(self, size=3):
        super().__init__()
        self.sp_conv = nn.Conv2d(96, 2, kernel_size=size, padding=size // 2)
        self.tp_conv = nn.Conv1d(3072, 2, kernel_size=size, padding=size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param (B, 96, T, 4, 8)
        @return (B, T, 3072)
        """
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, T, 96, 4, 8)
        
        '''spatial attention'''
        sp_x = x.view(-1, C, H, W)                      # (B*T, 96, 4, 8)

        sp_attention = self.sp_conv(sp_x)               # (B*T, 3, 4, 8)
        sp_attention1 = sp_attention[:, 0].unsqueeze(1) 
        sp_attention2 = sp_attention[:, 1].unsqueeze(1)
        
        sp_attention1 = self.sigmoid(sp_attention1)     # (B*T, 1, 4, 8)      
        sp_attention2 = self.sigmoid(sp_attention2)
        
        sp_attention1 = sp_attention1 * sp_x            # (B*T, 96, 4, 8)
        sp_attention2 = sp_attention2 * sp_x
        
        sp_attention1 = sp_attention1.view(B, T, -1)    # (B, T, 3072)
        sp_attention2 = sp_attention2.view(B, T, -1)

        '''temporal attention'''
        tp_x = x.view(B, T, -1)                         # (B, T, 3072)
        tp_x = tp_x.permute(0, 2, 1).contiguous()       # (B, 3072, T)
        
        tp_attention = self.tp_conv(tp_x)               # (B, 3, T)
        tp_attention1 = tp_attention[:, 0].unsqueeze(1) 
        tp_attention2 = tp_attention[:, 1].unsqueeze(1)

        tp_attention1 = self.sigmoid(tp_attention1)     # (B, 1, T)
        tp_attention2 = self.sigmoid(tp_attention2)
        
        tp_attention1 = tp_attention1 * tp_x            # (B, 3072, T)
        tp_attention2 = tp_attention2 * tp_x

        tp_attention1 = tp_attention1.permute(0, 2, 1).contiguous()
        tp_attention2 = tp_attention2.permute(0, 2, 1).contiguous()
        
        '''add norm'''
        feature1 = F.normalize(sp_attention1 + tp_attention1, p=2, dim=2)
        feature2 = F.normalize(sp_attention2 + tp_attention2, p=2, dim=2)
        
        feature = feature1 + feature2
        return feature
    
class SATA(nn.Module):
    """
    separate Spatial Temporal Attention
    """
    def __init__(self, in_channel=96):
        super().__init__()
        self.sp_conv = nn.Conv2d(in_channel, out_channels=3, kernel_size=3, padding=1)
        self.tp_conv = nn.Conv1d(in_channel * 4 * 8, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param (B, 96, T, 4, 8)
        @return (B, T, 3072)
        """
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, T, 96, 4, 8)
        
        '''spatial attention'''
        sp_x = x.view(-1, C, H, W)                      # (B*T, 96, 4, 8)

        sp_attention = self.sp_conv(sp_x)               # (B*T, 3, 4, 8)
        sp_attention1 = sp_attention[:, 0].unsqueeze(1) 
        sp_attention2 = sp_attention[:, 1].unsqueeze(1)
        sp_attention3 = sp_attention[:, 2].unsqueeze(1)
        
        sp_attention1 = self.sigmoid(sp_attention1)     # (B*T, 1, 4, 8)      
        sp_attention2 = self.sigmoid(sp_attention2)
        sp_attention3 = self.sigmoid(sp_attention3)
        
        sp_attention1 = sp_attention1 * sp_x            # (B*T, 96, 4, 8)
        sp_attention2 = sp_attention2 * sp_x
        sp_attention3 = sp_attention3 * sp_x
        
        sp_attention1 = sp_attention1.view(B, T, -1)    # (B, T, 3072)
        sp_attention2 = sp_attention2.view(B, T, -1)
        sp_attention3 = sp_attention3.view(B, T, -1)

        '''temporal attention'''
        tp_x = x.view(B, T, -1)                         # (B, T, 3072)
        tp_x = tp_x.permute(0, 2, 1).contiguous()       # (B, 3072, T)
        
        tp_attention = self.tp_conv(tp_x)               # (B, 3, T)
        tp_attention1 = tp_attention[:, 0].unsqueeze(1) 
        tp_attention2 = tp_attention[:, 1].unsqueeze(1)
        tp_attention3 = tp_attention[:, 2].unsqueeze(1)

        tp_attention1 = self.sigmoid(tp_attention1)     # (B, 1, T)
        tp_attention2 = self.sigmoid(tp_attention2)
        tp_attention3 = self.sigmoid(tp_attention3)
        
        tp_attention1 = tp_attention1 * tp_x            # (B, 3072, T)
        tp_attention2 = tp_attention2 * tp_x
        tp_attention3 = tp_attention3 * tp_x

        tp_attention1 = tp_attention1.permute(0, 2, 1).contiguous()
        tp_attention2 = tp_attention2.permute(0, 2, 1).contiguous()
        tp_attention3 = tp_attention3.permute(0, 2, 1).contiguous()
        
        '''add norm'''
        feature1 = F.normalize(sp_attention1 + tp_attention1, p=2, dim=2)
        feature2 = F.normalize(sp_attention2 + tp_attention2, p=2, dim=2)
        feature3 = F.normalize(sp_attention3 + tp_attention3, p=2, dim=2)
        
        feature = feature1 + feature2 + feature3
        return feature

class SATA4(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.sp_conv = nn.Conv2d(96, 4, kernel_size=size, padding=size // 2)
        self.tp_conv = nn.Conv1d(3072, 4, kernel_size=size, padding=size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param (B, 96, T, 4, 8)
        @return (B, T, 3072)
        """
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, T, 96, 4, 8)
        
        '''spatial attention'''
        sp_x = x.view(-1, C, H, W)                      # (B*T, 96, 4, 8)

        sp_attention = self.sp_conv(sp_x)               # (B*T, 3, 4, 8)
        sp_attention1 = sp_attention[:, 0].unsqueeze(1) 
        sp_attention2 = sp_attention[:, 1].unsqueeze(1)
        sp_attention3 = sp_attention[:, 2].unsqueeze(1)
        sp_attention4 = sp_attention[:, 3].unsqueeze(1)
        
        sp_attention1 = self.sigmoid(sp_attention1)     # (B*T, 1, 4, 8)      
        sp_attention2 = self.sigmoid(sp_attention2)
        sp_attention3 = self.sigmoid(sp_attention3)
        sp_attention4 = self.sigmoid(sp_attention4)
        
        sp_attention1 = sp_attention1 * sp_x            # (B*T, 96, 4, 8)
        sp_attention2 = sp_attention2 * sp_x
        sp_attention3 = sp_attention3 * sp_x
        sp_attention4 = sp_attention4 * sp_x
        
        sp_attention1 = sp_attention1.view(B, T, -1)    # (B, T, 3072)
        sp_attention2 = sp_attention2.view(B, T, -1)
        sp_attention3 = sp_attention3.view(B, T, -1)
        sp_attention4 = sp_attention4.view(B, T, -1)

        '''temporal attention'''
        tp_x = x.view(B, T, -1)                         # (B, T, 3072)
        tp_x = tp_x.permute(0, 2, 1).contiguous()       # (B, 3072, T)
        
        tp_attention = self.tp_conv(tp_x)               # (B, 3, T)
        tp_attention1 = tp_attention[:, 0].unsqueeze(1) 
        tp_attention2 = tp_attention[:, 1].unsqueeze(1)
        tp_attention3 = tp_attention[:, 2].unsqueeze(1)
        tp_attention4 = tp_attention[:, 3].unsqueeze(1)

        tp_attention1 = self.sigmoid(tp_attention1)     # (B, 1, T)
        tp_attention2 = self.sigmoid(tp_attention2)
        tp_attention3 = self.sigmoid(tp_attention3)
        tp_attention4 = self.sigmoid(tp_attention4)
        
        tp_attention1 = tp_attention1 * tp_x            # (B, 3072, T)
        tp_attention2 = tp_attention2 * tp_x
        tp_attention3 = tp_attention3 * tp_x
        tp_attention4 = tp_attention4 * tp_x

        tp_attention1 = tp_attention1.permute(0, 2, 1).contiguous()
        tp_attention2 = tp_attention2.permute(0, 2, 1).contiguous()
        tp_attention3 = tp_attention3.permute(0, 2, 1).contiguous()
        tp_attention4 = tp_attention4.permute(0, 2, 1).contiguous()
        
        '''add norm'''
        feature1 = F.normalize(sp_attention1 + tp_attention1, p=2, dim=2)
        feature2 = F.normalize(sp_attention2 + tp_attention2, p=2, dim=2)
        feature3 = F.normalize(sp_attention3 + tp_attention3, p=2, dim=2)
        feature4 = F.normalize(sp_attention4 + tp_attention4, p=2, dim=2)
        
        feature = feature1 + feature2 + feature3 + feature4
        return feature

class SATA5(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.sp_conv = nn.Conv2d(96, 5, kernel_size=size, padding=size // 2)
        self.tp_conv = nn.Conv1d(3072, 5, kernel_size=size, padding=size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param (B, 96, T, 4, 8)
        @return (B, T, 3072)
        """
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, T, 96, 4, 8)
        
        '''spatial attention'''
        sp_x = x.view(-1, C, H, W)                      # (B*T, 96, 4, 8)

        sp_attention = self.sp_conv(sp_x)               # (B*T, 3, 4, 8)
        sp_attention1 = sp_attention[:, 0].unsqueeze(1) 
        sp_attention2 = sp_attention[:, 1].unsqueeze(1)
        sp_attention3 = sp_attention[:, 2].unsqueeze(1)
        sp_attention4 = sp_attention[:, 3].unsqueeze(1)
        sp_attention5 = sp_attention[:, 4].unsqueeze(1)
        
        sp_attention1 = self.sigmoid(sp_attention1)     # (B*T, 1, 4, 8)      
        sp_attention2 = self.sigmoid(sp_attention2)
        sp_attention3 = self.sigmoid(sp_attention3)
        sp_attention4 = self.sigmoid(sp_attention4)
        sp_attention5 = self.sigmoid(sp_attention5)
        
        sp_attention1 = sp_attention1 * sp_x            # (B*T, 96, 4, 8)
        sp_attention2 = sp_attention2 * sp_x
        sp_attention3 = sp_attention3 * sp_x
        sp_attention4 = sp_attention4 * sp_x
        sp_attention5 = sp_attention5 * sp_x
        
        sp_attention1 = sp_attention1.view(B, T, -1)    # (B, T, 3072)
        sp_attention2 = sp_attention2.view(B, T, -1)
        sp_attention3 = sp_attention3.view(B, T, -1)
        sp_attention4 = sp_attention4.view(B, T, -1)
        sp_attention5 = sp_attention5.view(B, T, -1)

        '''temporal attention'''
        tp_x = x.view(B, T, -1)                         # (B, T, 3072)
        tp_x = tp_x.permute(0, 2, 1).contiguous()       # (B, 3072, T)
        
        tp_attention = self.tp_conv(tp_x)               # (B, 3, T)
        tp_attention1 = tp_attention[:, 0].unsqueeze(1) 
        tp_attention2 = tp_attention[:, 1].unsqueeze(1)
        tp_attention3 = tp_attention[:, 2].unsqueeze(1)
        tp_attention4 = tp_attention[:, 3].unsqueeze(1)
        tp_attention5 = tp_attention[:, 4].unsqueeze(1)

        tp_attention1 = self.sigmoid(tp_attention1)     # (B, 1, T)
        tp_attention2 = self.sigmoid(tp_attention2)
        tp_attention3 = self.sigmoid(tp_attention3)
        tp_attention4 = self.sigmoid(tp_attention4)
        tp_attention5 = self.sigmoid(tp_attention5)
        
        tp_attention1 = tp_attention1 * tp_x            # (B, 3072, T)
        tp_attention2 = tp_attention2 * tp_x
        tp_attention3 = tp_attention3 * tp_x
        tp_attention4 = tp_attention4 * tp_x
        tp_attention5 = tp_attention5 * tp_x

        tp_attention1 = tp_attention1.permute(0, 2, 1).contiguous()
        tp_attention2 = tp_attention2.permute(0, 2, 1).contiguous()
        tp_attention3 = tp_attention3.permute(0, 2, 1).contiguous()
        tp_attention4 = tp_attention4.permute(0, 2, 1).contiguous()
        tp_attention5 = tp_attention5.permute(0, 2, 1).contiguous()

        '''add norm'''
        feature1 = F.normalize(sp_attention1 + tp_attention1, p=2, dim=2)
        feature2 = F.normalize(sp_attention2 + tp_attention2, p=2, dim=2)
        feature3 = F.normalize(sp_attention3 + tp_attention3, p=2, dim=2)
        feature4 = F.normalize(sp_attention4 + tp_attention4, p=2, dim=2)
        feature5 = F.normalize(sp_attention5 + tp_attention5, p=2, dim=2)
        
        feature = feature1 + feature2 + feature3 + feature4 + feature5
        return feature

if __name__ == '__main__':
    model = SATA()
    x = torch.rand(4, 96, 10, 4, 8)
    y = model(x)
    print(y.shape)