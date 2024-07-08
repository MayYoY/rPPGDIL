from collections import OrderedDict
import torch
import torch.nn as nn
import os
import copy
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from models.peft import custom_adapter

model_path = ''
model_path = {
    'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
}


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=QuickGELU, drop=0.,
                 adapter_cls=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if adapter_cls:
            self.adapter = adapter_cls(D_features=in_features, skip_connect=False)
            self.scale = 0.5
        else:
            self.adapter = None

    def forward(self, x):
        if self.adapter:
            residual = self.adapter(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.adapter:
            x += residual
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 adapter_cls=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if adapter_cls:
            self.adapter = adapter_cls(D_features=dim, skip_connect=False)
        else:
            self.adapter = None

    def forward(self, x, name=None):
        B, N, C = x.shape
        if self.adapter:
            residual = self.adapter(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.adapter:
            x += residual
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=QuickGELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=QuickGELU, norm_layer=nn.LayerNorm, peft_cls=None):
        super().__init__()
        self.dim = dim
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.conv_pefts = nn.ModuleDict()
        self.mlp_pefts = nn.ModuleDict()
        if peft_cls:
            # self.conv_pefts.add_module("task0", peft_cls(dim, skip_connect=False))
            self.mlp_pefts.add_module("task0", peft_cls(dim, skip_connect=False))

    def add_peft(self, peft_cls, names, peft_module=None):
        for name in names:
            # self.conv_pefts.add_module(name, peft_cls(self.dim, skip_connect=False))
            self.mlp_pefts.add_module(name, peft_cls(self.dim, skip_connect=False))

    def attach_peft(self, peft_module, name):
        # self.conv_pefts[name] = copy.copy(peft_module)  # NOTE: shallow copy
        self.mlp_pefts[name] = copy.copy(peft_module)

    def freeze_peft(self, names):
        for name, param in self.conv_pefts.named_parameters():
            if name in names:
                param.requires_grad = False
        for name, param in self.mlp_pefts.named_parameters():
            if name in names:
                param.requires_grad = False

    def forward(self, x, name="task0"):
        x = x + self.pos_embed(x)

        if len(self.conv_pefts) and name:
            # print("conv_peft")
            x = self.conv_pefts[name](self.norm1(x)) + x + self.drop_path(
                self.conv2(self.attn(self.conv1(self.norm1(x)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        if len(self.mlp_pefts) and name:
            # print("mlp_peft")
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.mlp_pefts[name](self.norm2(x))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=QuickGELU, norm_layer=nn.LayerNorm, peft_cls=None):
        super().__init__()
        self.dim = dim
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn_pefts = nn.ModuleDict()
        self.mlp_pefts = nn.ModuleDict()
        if peft_cls:
            # self.attn_pefts.add_module("task0", peft_cls(dim, skip_connect=False))
            self.mlp_pefts.add_module("task0", peft_cls(dim, skip_connect=False))

    def add_peft(self, peft_cls, names):
        for name in names:
            # self.attn_pefts.add_module(name, peft_cls(self.dim, skip_connect=False))
            self.mlp_pefts.add_module(name, peft_cls(self.dim, skip_connect=False))

    def attach_peft(self, peft_module, name):
        # self.attn_pefts[name] = copy.copy(peft_module)  # NOTE: shallow copy
        self.mlp_pefts[name] = copy.copy(peft_module)

    def freeze_peft(self, names):
        for name, param in self.attn_pefts.named_parameters():
            if name in names:
                param.requires_grad = False
        for name, param in self.mlp_pefts.named_parameters():
            if name in names:
                param.requires_grad = False

    def forward(self, x, name="task0"):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if len(self.attn_pefts) and name:
            # print("attn_peft")
            x = self.attn_pefts[name](self.norm1(x)) + x + self.drop_path(self.attn(self.norm1(x), name=name))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), name=name))
        if len(self.mlp_pefts) and name:
            # print("mlp_peft")
            x = x + self.drop_path(self.mlp(self.norm2(x))) + self.mlp_pefts[name](self.norm2(x))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=QuickGELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Uniformer(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], num_classes=400, img_size=224, in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.3, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False,
                 conv_peft=None, attn_peft=None, pool_size=0):
        super().__init__()

        self.num_classes = num_classes
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, peft_cls=conv_peft)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0]], norm_layer=norm_layer,
                peft_cls=conv_peft)
            for i in range(depth[1])])
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer)
                for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer, peft_cls=attn_peft,
                )
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer, peft_cls=attn_peft,
                )
                for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])

        self.pool_size = pool_size
        if pool_size:
            self.peft_key = nn.Parameter(torch.FloatTensor(pool_size, embed_dim[-1]), requires_grad=True)
            self.peft_names = [f"task{i}" for i in range(pool_size)]
            self.add_peft(conv_peft, attn_peft, self.peft_names)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

        # NOTE: initialize pefts
        for n2, m2 in self.named_modules():
            if isinstance(m2, custom_adapter.AimAdapter) or isinstance(m2, custom_adapter.Conv3dAdapter):
                m2.init_peft()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def add_peft(self, conv_peft=custom_adapter.Conv3dAdapter, attn_peft=custom_adapter.AimAdapter, names=None):
        for blk in self.blocks1:
            blk.add_peft(conv_peft, names)
        for blk in self.blocks2:
            blk.add_peft(conv_peft, names)
        for blk in self.blocks3:
            blk.add_peft(attn_peft, names)
        for blk in self.blocks4:
            blk.add_peft(attn_peft, names)

    def get_pretrained_model(self, pretrain_name="uniformer_small_k400_16x4"):
        if pretrain_name:
            checkpoint = torch.load(model_path[pretrain_name], map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            del checkpoint['head.weight']
            del checkpoint['head.bias']

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        continue
                    print(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)
            return checkpoint
        else:
            return None

    def freeze_model(self):
        for name, param in self.named_parameters():
            if 'peft' not in name:
                param.requires_grad = False

    def freeze_peft(self, peft_name):
        for name, param in self.named_parameters():
            if 'peft' in name and peft_name in name:
                param.requires_grad = False

    def unfreeze_peft(self, peft_name):
        for name, param in self.named_parameters():
            if 'peft' in name and peft_name in name:
                param.requires_grad = True

    def get_conv1_feature(self, x, name="task0"):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x, name=name)
        return x

    def conv1_to_conv2_feature(self, x, name="task0"):
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, name=name)
        return x

    def get_conv_feature(self, x, name="task0"):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x, name=name)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, name=name)
        return x

    def forward_conv_feature(self, x, name="task0"):
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x, name=name)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x, name=name)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward_features(self, x, name="task0"):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x, name=name)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, name=name)
        conv_feature = x.detach()  # Conv feature
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x, name=name)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x, name=name)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x, conv_feature

    def forward(self, x, name="task0", ret_conv=False):
        x, conv_feature = self.forward_features(x, name=name)
        if ret_conv:
            return x, conv_feature
        return x


def uniformer_small(peft=None):
    if peft == "adapter":
        conv_peft = custom_adapter.Conv3dAdapter
        attn_peft = custom_adapter.AimAdapter
    else:
        conv_peft = None
        attn_peft = None
    return Uniformer(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512],
        head_dim=64, drop_rate=0.1, conv_peft=conv_peft, attn_peft=attn_peft)


def uniformer_base():
    return Uniformer(
        depth=[5, 8, 20, 7], embed_dim=[64, 128, 320, 512],
        head_dim=64, drop_rate=0.3)
