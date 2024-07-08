import torch
import torch.nn as nn
import math


class AimAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.peft_scale = nn.Parameter(torch.tensor(0.5))
        self.init_peft()

    def init_peft(self):
        for n2, m2 in self.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Linear):
                    nn.init.constant_(m2.weight, 0)
                    nn.init.constant_(m2.bias, 0)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x * self.peft_scale
    

class Conv3dAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Conv3d(D_features, D_hidden_features, kernel_size=1)
        self.D_fc2 = nn.Conv3d(D_hidden_features, D_features, kernel_size=1)
        self.peft_scale = nn.Parameter(torch.tensor(0.5))
        self.init_peft()

    def init_peft(self):
        for n2, m2 in self.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Conv3d):
                    nn.init.constant_(m2.weight, 0)
                    nn.init.constant_(m2.bias, 0)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x * self.peft_scale


if __name__ == '__main__':
    config = None
