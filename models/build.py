import torch
import torch.nn as nn
from copy import deepcopy
from .backbone import uniformer
from . import heads, diffnorm
from tools import logging


logger = logging.get_logger(__name__)


class RPPGModel(nn.Module):
    def __init__(self, backbone=nn.Identity(), head=nn.Identity(),
                 use_diffnorm=False, head_pool_size=1):
        super(RPPGModel, self).__init__()
        if use_diffnorm:
            self.dn = diffnorm.DiffNorm()
        self.use_diffnorm = use_diffnorm
        self.backbone = backbone

        self.head_pool_size = head_pool_size
        if head_pool_size > 1:
            self.regressor = nn.ModuleDict({"task0": head})
        else:
            self.regressor = head
    
    def add_peft(self, conv_peft, attn_peft, names):
        self.backbone.add_peft(conv_peft, attn_peft, names)

    def add_head(self, task_idx):
        assert self.head_pool_size > 1, "No need to attach head"
        self.regressor.add_module(f"task{task_idx}", deepcopy(self.regressor[f"task{task_idx - 1}"]))

    def freeze_head(self, head_name):
        for name, param in self.regressor[head_name].named_parameters():
            param.requires_grad = False
    
    def unfreeze_head(self, head_name):
        for name, param in self.regressor[head_name].named_parameters():
            param.requires_grad = True
    
    def freeze_model(self):
        for name, param in self.named_parameters():
            if 'peft' not in name:
                param.requires_grad = False

    def get_conv1_feature(self, x, name="task0"):
        if self.use_diffnorm:
            x = self.dn(x)
        x = self.backbone.get_conv1_feature(x, name=name)
        return x
    
    def conv1_to_conv2_feature(self, x, name="task0"):
        x = self.backbone.conv1_to_conv2_feature(x, name=name)
        return x

    def get_conv_feature(self, x, name="task0"):
        if self.use_diffnorm:
            x = self.dn(x)
        x = self.backbone.get_conv_feature(x, name=name)
        return x

    def forward_conv_feature(self, x, name="task0"):
        x = self.backbone.forward_conv_feature(x, name=name)
        if self.head_pool_size > 1 and name:
            x = self.regressor[name](x)
        else:
            x = self.regressor(x)
        return x
    
    def conv_to_regress(self, x, name="task0", avg_pool=True):
        x = self.backbone.forward_conv_feature(x, name=name)
        if avg_pool:
            return x.flatten(2).mean(-1)  # B C
        else:
            return x

    def extract_feature(self, x, name="task0", avg_pool=True):
        if self.use_diffnorm:
            x = self.dn(x)
        x = self.backbone(x, name)
        if avg_pool:
            return x.flatten(2).mean(-1)  # B C
        else:
            return x
        
    def forward_regress_feature(self, x, name="task0"):
        if self.head_pool_size > 1 and name:
            x = self.regressor[name](x)
        else:
            x = self.regressor(x)
        return x

    def forward(self, x, name="task0"):
        if self.use_diffnorm:
            x = self.dn(x)
        x = self.backbone(x, name)  # b c t h w
        if self.head_pool_size > 1 and name:
            x = self.regressor[name](x)
        else:
            x = self.regressor(x)
        return x


def build_model(cfg=None, gpu_id=None):
    if torch.cuda.is_available():
        assert (
            cfg.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.num_gpus == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs." 

    print(f"Uniformer Small ! DiffNorm {cfg.diffnorm}")
    backbone = uniformer.uniformer_small(cfg.peft)
    regress_head = heads.UpRegressor(dim=512)
    
    if cfg.pretrain:
        logger.info('load pretrained model')
        checkpoint = backbone.get_pretrained_model()
        msg = backbone.load_state_dict(checkpoint, strict=False)
        logger.info(f"Missing Keys: {msg.missing_keys}")
        logger.info(f"Unexpected Keys: {msg.unexpected_keys}")
        torch.cuda.empty_cache()
    if cfg.freeze:
        logger.info('freeze pretrained weights')
        backbone.freeze_model()

    if hasattr(cfg, "head_pool_size"):
        model = RPPGModel(backbone=backbone, head=regress_head, use_diffnorm=cfg.diffnorm, head_pool_size=cfg.head_pool_size)
    else:
        model = RPPGModel(backbone=backbone, head=regress_head, use_diffnorm=cfg.diffnorm)
    if cfg.continual_pretrain is not None:
        msg = model.load_state_dict(torch.load(cfg.continual_pretrain, map_location="cpu"), False)
        logger.info(f"Missing Keys: {msg.missing_keys}")
        logger.info(f"Unexpected Keys: {msg.unexpected_keys}")
        if cfg.freeze:
            model.freeze_model()
            logger.info(f"Freeze all modules except peft")

    if cfg.num_gpus:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.num_gpus > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=False
        )
    return model
