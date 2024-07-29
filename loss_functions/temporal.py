import torch
import torch.nn as nn


class NegPearson(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(NegPearson, self).__init__()
        assert reduction in ["mean", "sum", "none"], "Unsupported reduction type!"
        self.reduction = reduction

    def forward(self, preds, labels):
        with torch.cuda.amp.autocast(enabled=False):
            preds = preds.to(torch.float32)
            labels = labels.to(torch.float32)
            sum_x = torch.sum(preds, dim=1)
            sum_y = torch.sum(labels, dim=1)
            sum_xy = torch.sum(labels * preds, dim=1)
            sum_x2 = torch.sum(preds ** 2, dim=1)
            sum_y2 = torch.sum(labels ** 2, dim=1)
            T = preds.shape[1]
            denominator = (T * sum_x2 - sum_x ** 2) * (T * sum_y2 - sum_y ** 2)
            for i in range(len(denominator)):
                denominator[i] = max(denominator[i], 1e-6)
            loss = 1 - ((T * sum_xy - sum_x * sum_y) / (torch.sqrt(denominator) + 1e-6))
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
        return loss
