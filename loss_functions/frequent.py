import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def normal_sampling(mean, label_k, std=1.0):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


class FreqLoss(nn.Module):
    def __init__(self, T=300, delta=3, reduction="mean"):
        super(FreqLoss, self).__init__()
        self.T = T
        self.delta = delta
        self.low_bound = 40
        self.high_bound = 180
        # for DFT
        self.bpm_range = torch.arange(self.low_bound, self.high_bound,
                                      dtype=torch.float) / 60.
        self.two_pi_n = Variable(2 * math.pi * torch.arange(0, self.T, dtype=torch.float))
        self.hanning = Variable(torch.from_numpy(np.hanning(self.T)).type(torch.FloatTensor),
                                requires_grad=True).view(1, -1)  # 1 x N
        # criterion
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_diverse = nn.KLDivLoss(reduction="sum")
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, wave, labels, fps, raw_mae=False):
        self.bpm_range = self.bpm_range.to(wave.device)
        self.two_pi_n = self.two_pi_n.to(wave.device)
        self.hanning = self.hanning.to(wave.device)

        B = wave.shape[0]
        k = self.bpm_range[None, :] / fps[:, None]
        k = k.view(B, -1, 1)  # B x range x 1
        preds = wave * self.hanning  # B x N
        preds = preds.view(B, 1, -1)  # B x 1 x N
        temp = self.two_pi_n.repeat(B, 1)
        temp = temp.view(B, 1, -1)  # B x 1 x N
        # B x range
        with torch.cuda.amp.autocast(enabled=False):
            preds = preds.to(torch.float32)
            k = k.to(torch.float32)
            temp = temp.to(torch.float32)
            complex_absolute = torch.sum(preds * torch.sin(k * temp), dim=-1) ** 2 \
                               + torch.sum(preds * torch.cos(k * temp), dim=-1) ** 2
            norm_t = torch.ones(B, device=wave.device, dtype=torch.float32) / (
                        torch.sum(complex_absolute, dim=1) + 1e-6)
        with torch.cuda.amp.autocast(enabled=True):
            norm_t = norm_t.view(-1, 1)  # B x 1
            complex_absolute = complex_absolute * norm_t  # B x range
            gts = labels.clone()
            gts -= self.low_bound
            gts[gts.le(0)] = 0
            gts[gts.ge(139)] = 139
            gts = gts.type(torch.long).view(B)

            _, whole_max_idx = complex_absolute.max(1)
            freq_loss = self.cross_entropy(complex_absolute, gts)

            gts_distribution = []
            for gt in gts:
                temp = [normal_sampling(int(gt), i, std=1.) for i in range(140)]
                temp = [i if i > 1e-6 else 1e-6 for i in temp]
                gts_distribution.append(temp)
            gts_distribution = torch.tensor(gts_distribution, device=wave.device)
            freq_distribution = F.log_softmax(complex_absolute, dim=-1)
            dist_loss = self.kl_diverse(freq_distribution, gts_distribution) / B

            # MAE loss
            mae_loss = self.l1_loss(whole_max_idx.type(torch.float), gts.type(torch.float))
        if not raw_mae:
            return dist_loss, freq_loss, mae_loss
        else:
            return dist_loss, freq_loss, mae_loss, torch.abs(whole_max_idx.type(torch.float) - gts.type(torch.float))


class FreqLogit(nn.Module):
    def __init__(self, T=300):
        super(FreqLogit, self).__init__()
        self.T = T
        self.low_bound = 40
        self.high_bound = 180
        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float) / 60.
        self.two_pi_n = Variable(2 * math.pi * torch.arange(0, self.T, dtype=torch.float))
        self.hanning = Variable(torch.from_numpy(np.hanning(self.T)).type(torch.FloatTensor),
                                requires_grad=True).view(1, -1)  # 1 x N

    def forward(self, wave, fps):
        self.bpm_range = self.bpm_range.to(wave.device)
        self.two_pi_n = self.two_pi_n.to(wave.device)
        self.hanning = self.hanning.to(wave.device)

        # DFT
        B = wave.shape[0]
        k = self.bpm_range[None, :] / fps[:, None]
        k = k.view(B, -1, 1)  # B x range x 1
        preds = wave * self.hanning  # B x N
        preds = preds.view(B, 1, -1)  # B x 1 x N
        # 2 \pi n
        temp = self.two_pi_n.repeat(B, 1)
        temp = temp.view(B, 1, -1)  # B x 1 x N
        # B x range
        with torch.cuda.amp.autocast(enabled=False):
            preds = preds.to(torch.float32)
            k = k.to(torch.float32)
            temp = temp.to(torch.float32)
            complex_absolute = torch.sum(preds * torch.sin(k * temp), dim=-1) ** 2 \
                               + torch.sum(preds * torch.cos(k * temp), dim=-1) ** 2
        return complex_absolute
