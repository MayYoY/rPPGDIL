import torch
import torch.nn as nn


class DiffNorm(nn.Module):
    def __init__(self, in_chans=3) -> None:
        super().__init__()
        self.norm = nn.BatchNorm3d(in_chans)
        self.proj = nn.Conv3d(in_channels=in_chans * 2, out_channels=in_chans, kernel_size=3, padding=1)

    def freeze_model(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_diff = torch.cat([torch.zeros((B, C, 1, H, W), device=x.device), x.diff(dim=2)], dim=2)
        x_diff = self.norm(x_diff)
        x = torch.cat([x, x_diff], dim=1)
        return self.proj(x)
