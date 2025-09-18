import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConsistencyLoss(nn.Module):
    def __init__(self, use_strategy_weighting=False):
        super().__init__()
        self.use_strategy_weighting = use_strategy_weighting
        
    def forward(self, feat_anchor, feat_positive, weights=None):
        cos_sim = F.cosine_similarity(feat_anchor, feat_positive)
        loss = 1 - cos_sim
        
        if self.use_strategy_weighting and weights is not None:
            loss = loss * weights
        
        return torch.mean(loss)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, use_strategy_weighting=False):
        super().__init__()
        self.temperature = temperature
        self.use_strategy_weighting = use_strategy_weighting
        
    def forward(self, features, labels, weights=None):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(1, 1)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        
        if self.use_strategy_weighting and weights is not None:
            loss = loss * weights
        
        return loss.mean()