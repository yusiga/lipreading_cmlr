import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(MultiCrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *args):
        # (predict, target)
        loss = None
        for i in range(len(args)):
            tmp = self.criteron(*args[i])
            if loss is None:
                loss = tmp
            else:
                loss += tmp
        return loss


class MultiWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(MultiWeightedCrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *args):
        # (predict, target, weight)
        loss = None
        for i in range(len(args)):
            predict, target, weight = args[i]
            tmp = weight * self.criteron(predict, target)
            if loss is None:
                loss = tmp
            else:
                loss += tmp
        return loss

class UnionLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(UnionLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse = nn.MSELoss()

    def forward(self, predict_pinyin, target_pinyin, weight_pinyin, \
                predict_hanzi, target_hanzi, weight_hanzi, \
                predict_mapped, target_mapped, weight_mapped):
        loss = weight_pinyin * self.ce(predict_pinyin, target_pinyin)
        loss += weight_hanzi * self.ce(predict_hanzi, target_hanzi)
        loss += weight_mapped * self.mse(predict_mapped, target_mapped)
        return loss

def kl_loss(prior_out, posterior_out,model,lambda_l2=0.01):
        # 确保张量在 CUDA 上
        if prior_out.is_cuda:
            prior_out = prior_out.cuda()
        if posterior_out.is_cuda:
            posterior_out = posterior_out.cuda()
        # 检查输入值是否有 nan 或 inf
        if torch.isnan(prior_out).any() or torch.isinf(prior_out).any():
            raise ValueError("prior_out contains NaN or Inf values")
        if torch.isnan(posterior_out).any() or torch.isinf(posterior_out).any():
            raise ValueError("posterior_out contains NaN or Inf values")
        
        # 归一化输入以防止数值溢出或下溢
        # prior_out_norm = torch.norm(prior_out, dim=-1, keepdim=True) + 1e-8
        # posterior_out_norm = torch.norm(posterior_out, dim=-1, keepdim=True) + 1e-8
        # prior_out = prior_out / prior_out_norm
        # posterior_out = posterior_out / posterior_out_norm
        
        kl_1 = F.kl_div(prior_out.log_softmax(-1), posterior_out.softmax(-1), reduction="sum") # sum
        # print(f'kl_1:{kl_1}')
        kl_2 = F.kl_div(posterior_out.log_softmax(-1), prior_out.softmax(-1), reduction="sum")
        # print(f'kl_2:{kl_2}')
        kl_loss = (kl_1 + kl_2) / 2
        # print(f'kl_loss:{kl_loss}')
        # Calculate L2 regularization (weight decay)
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        # Combine KL divergence loss and L2 regularization
        kl_loss = kl_loss + lambda_l2 * l2_reg
        return kl_loss
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.cosine_embedding_loss(anchor, positive, torch.ones(anchor.size(0)).to(anchor.device))
    distance_negative = F.cosine_embedding_loss(anchor, negative, torch.zeros(anchor.size(0)).to(anchor.device))
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()
    