import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Regressor(nn.Module):
    def __init__(self, dim=768):
        super(Regressor, self).__init__()
        self.spat = nn.AdaptiveAvgPool2d((1, 1))
        self.temp1 = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2),
            QuickGELU()
        )
        self.temp2 = nn.Sequential(
            nn.Conv1d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2),
            QuickGELU()
        )
        self.regress = nn.Conv1d(in_channels=dim // 2, out_channels=1, kernel_size=1)

    def freeze_model(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        # B C T H W
        # b, c, t, _, _ = x.shape
        # x = x.view(b, c, t)
        x = self.spat(x).flatten(2)
        x = self.temp2(self.temp1(x))
        return self.regress(x).squeeze(1)


class UpRegressor(nn.Module):
    def __init__(self, dim=512, ratio=2):
        super(UpRegressor, self).__init__()
        self.spat = nn.AdaptiveAvgPool2d((1, 1))
        # k = 2p + s
        self.temp = nn.Sequential(
            nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=4, padding=1, stride=2),
            nn.Conv1d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2),
            QuickGELU()
        )
        self.regress = nn.Conv1d(in_channels=dim // 2, out_channels=1, kernel_size=1)
        
    def freeze_model(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # B C T H W
        if len(x.shape) == 5:
            x = self.spat(x).flatten(2)
        x = self.temp(x)  # b c t
        return self.regress(x).squeeze(1)
