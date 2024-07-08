import torch
import torch.nn as nn
from torch import distributed as dist
import math
import os
from tqdm.auto import tqdm
import numpy as np
from timm.scheduler import CosineLRScheduler

from models import build
from tools import distributed as mydist
from tools import logging
from loss_functions import temporal, frequent
from evaluate import metrics
from scripts import test
from evaluate import postprocess


logger = logging.get_logger(__name__)


class Naive:
    def __init__(self, train_config, debug=False):
        self.train_config = train_config
        self.net = build.build_model(train_config)
        self.optimizer = None
        self.scheduler = None
        self.data_iter = None
        self.data_sampler = None

        self.num_epochs = self.train_config.init_epoch
        self.loss_fun1 = temporal.NegPearson()
        self.loss_fun2 = frequent.FreqLoss(T=self.train_config.num_frames)  # CE + Distribution
        self.train_loss = metrics.Accumulate(4, names=["temporal loss", "ce loss", "dist loss", "mae"])  # for print

    def before_task(self, data_iter, data_sampler, task_idx):
        if task_idx:
            self.num_epochs = self.train_config.contin_epoch
            local_lr = self.train_config.contin_lr
        else:
            local_lr = self.train_config.init_lr
        self.data_iter = data_iter
        self.data_sampler = data_sampler
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=local_lr, 
                                          weight_decay=self.train_config.weight_decay, eps=1e-6)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.num_epochs, lr_min=1e-5, 
                                           warmup_t=self.train_config.warmup_t, warmup_lr_init=1e-5)

    def train_task(self, task_idx):
        self.net.train()
        self.net.cuda()
        epoch_timer = logging.EpochTimer()
        for epoch in range(self.num_epochs):
            if self.data_sampler:
                print("Sampler exist!!!")
                self.data_sampler.set_epoch(epoch)
            self.scheduler.step(epoch)  # !
            self.train_loss.reset()
            logger.info(f"========= Epoch {epoch + 1} =========")
            epoch_timer.epoch_tic()

            self.train_epoch(epoch, task_idx=task_idx)

            epoch_timer.epoch_toc()
            logger.info(
                f"Epoch {epoch + 1} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from epoch 1 to {epoch + 1} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
            )
            if epoch != self.num_epochs - 1:
                logger.info(
                    f"From epoch 1 to {epoch + 1}, "
                    f"{epoch_timer.sum_epoch_time():.2f}s used, "
                    f"about {epoch_timer.median_epoch_time() * (self.num_epochs - 1 - epoch):.2f}s remained."
                )
            if mydist.is_root_proc():
                logger.info(self.train_loss)
                if epoch == self.num_epochs - 1 or (epoch + 1) % 5 == 0:
                    os.makedirs(self.train_config.ckpt_path, exist_ok=True)
                    sd = self.net.module.state_dict() if self.train_config.num_gpus > 1 else self.net.state_dict()
                    torch.save(sd, self.train_config.ckpt_path + os.sep + f"{epoch + 1}.pt")
            mydist.synchronize()
    
    def post_task(self, task_idx):
        pass

    def train_epoch(self, epoch, task_idx=None):
        for batch_data in self.data_iter:
            x = batch_data["input"].cuda()
            y = batch_data["wave"].cuda()
            average_hr = batch_data["wave_hr"].view(-1).cuda()
            fs = batch_data["fs"].view(-1).cuda()

            self.before_iter()

            preds = self.net(x, name="task0")
            rppg_loss = self.loss_fun1(preds, y)
            dist_loss, freq_loss, mae_loss = self.loss_fun2(preds, average_hr, fs)
            a = 0.1 * math.pow(0.5, epoch / self.num_epochs)
            b = math.pow(5., epoch / self.num_epochs)
            loss = a * rppg_loss + b * (freq_loss + dist_loss)  # NOTE
            if loss.isnan().any():
                raise ValueError("NaN comes from total loss")

            self.optimizer.zero_grad()  # !
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            self.post_iter()
                
            if self.train_config.num_gpus > 1:
                [rppg_loss, freq_loss, dist_loss, mae_loss] = mydist.all_reduce([rppg_loss, freq_loss, dist_loss, mae_loss])
            val = [rppg_loss.data, freq_loss.data, dist_loss.data, mae_loss.data]
            self.train_loss.update(val=val, n=1)

    def before_iter(self):
        pass

    def post_iter(self):
        pass

    def inference(self, epoch, test_iter, task_idx=None):
        test.inference(self.train_config, epoch, verbal=False, test_iter=test_iter)

    def eval(self, epoch):
        test.eval(self.train_config, epoch, verbal=False)
