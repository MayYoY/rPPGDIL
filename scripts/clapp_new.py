import torch
import pprint
from torch import distributed as dist
import os
import random
import numpy as np

from models import build
from tools import distributed as mydist
from tools import logging
from loss_functions import temporal, frequent
from datasets import rppgdata
from evaluate import metrics
from configs import contintrain, contintest
import runner
from runner import mymethod

logger = logging.get_logger(__name__)
orders = [[0, 1, 2, 3, 4, 5]]


def fixSeed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class App:
    def __init__(self, config, args, mode="eval", task_order=0, upper_bound=False, debug=False) -> None:
        self.config = config
        self.task_order = task_order
        if debug:
            config.log_path = "./debug.log"
        if config.num_gpus > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(args.local_rank)
        elif config.num_gpus == 1:
            torch.cuda.set_device(config.gpu_id)

        self.task_names = ["Fold1-4", "PURE", "Fold5", "UBFC", "BUAA", "MMPD"]

        logging.setup_logging(config.log_path)
        logger.info("Train with config:")
        logger.info(pprint.pformat(config))
        logger.info("Task Order:")
        for i in range(6):
            logger.info(f"============= {self.task_names[orders[self.task_order][i]]} =============")

        fixSeed(42)  # NOTE
        if mode == "train":
            d0 = rppgdata.VIPLDataset(contintrain.ViplF14())
            d1 = rppgdata.UPDataset(contintrain.Pure())
            d2 = rppgdata.VIPLDataset(contintrain.ViplF5())
            d3 = rppgdata.UPDataset(contintrain.Ubfc())
            d4 = rppgdata.BUAADataset(contintrain.Buaa())
            d5 = rppgdata.MMPDDataset(contintrain.Mmpd_new())
        else:
            d0 = rppgdata.VIPLDataset(contintest.ViplF14())
            d1 = rppgdata.UPDataset(contintest.Pure())
            d2 = rppgdata.VIPLDataset(contintest.ViplF5())
            d3 = rppgdata.UPDataset(contintest.Ubfc())
            d4 = rppgdata.BUAADataset(contintest.Buaa())
            d5 = rppgdata.MMPDDataset(contintest.Mmpd_new())

        if mode == "train" and upper_bound:
            self.joint_dataset = torch.utils.data.ConcatDataset([d0, d1, d2, d3, d4, d5])
            self.joint_iter, self.joint_sampler = self.get_iter(self.joint_dataset, config.num_gpus, config.batch_size)
        else:
            self.datasets = [d0, d1, d2, d3, d4, d5]

        self.run = mymethod.MyMethod(config, debug=debug)
        # self.run = runner.Naive(config, debug=debug)

    def get_iter(self, ds, num_gpus, batch_size, shuffle=True):
        if num_gpus > 1:
            data_sampler = torch.utils.data.distributed.DistributedSampler(ds)
            data_iter = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                    shuffle=False, num_workers=4,
                                                    drop_last=True, sampler=data_sampler)
        else:
            data_sampler = None
            data_iter = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        logger.info(f"Length of train set {len(ds)}, train iter {len(data_iter)}")
        return data_iter, data_sampler

    def joint_train(self):
        logger.info(f"Joint training (upper bound) beginning...")
        self.run.before_task(self.joint_iter, self.joint_sampler, 0)
        self.run.train_task(0)
        self.run.post_task(0)

    def continual_train(self, num_tasks=6, start_task=0):
        temp = self.run.train_config.ckpt_path
        for idx in range(start_task, num_tasks):
            i = orders[self.task_order][idx]
            self.run.train_config.ckpt_path = temp + os.sep + self.task_names[i]
            logger.info(f"Task{idx} {self.task_names[i]} beginning...")
            task_iter, task_sampler = self.get_iter(self.datasets[i],
                                                    self.config.num_gpus,
                                                    self.config.batch_size,
                                                    shuffle=True)
            print(f"Length of train iter {len(task_iter)}")
            self.run.before_task(task_iter, task_sampler, idx)
            self.run.train_task(idx)
            self.run.post_task(idx)

    def continual_train_restar(self, restar_ckpt_path, start_task=1, num_tasks=6):
        assert start_task > 0
        last_task_idx = orders[self.task_order][start_task - 1]
        logger.info(f"First Resume Last Task: {self.task_names[last_task_idx]}")
        task_iter, task_sampler = self.get_iter(self.datasets[last_task_idx],
                                                self.config.num_gpus,
                                                self.config.batch_size,
                                                shuffle=True)
        temp = self.run.train_config.ckpt_path
        self.run.train_config.ckpt_path = temp + os.sep + self.task_names[last_task_idx]

        self.run.before_task(task_iter, task_sampler, start_task - 1)
        sd = torch.load(restar_ckpt_path, map_location="cpu")
        self.run.net.load_state_dict(sd)
        self.run.net.cuda()

        self.run.post_task(start_task - 1)

        self.run.train_config.ckpt_path = temp
        self.continual_train(num_tasks, start_task)

    def continual_inference(self, epoch=10):
        result_temp = self.run.train_config.result_path
        for task_idx, i in enumerate(orders[self.task_order]):
            self.run.train_config.result_path = result_temp + os.sep + self.task_names[i]
            logger.info(f"Task{task_idx} {self.task_names[i]} beginning...")
            task_iter, _ = self.get_iter(self.datasets[i], 1,
                                         self.config.batch_size,
                                         shuffle=False)
            if not isinstance(self.run, mymethod.MyMethod):
                self.run.inference(epoch, task_iter, task_idx=task_idx)
            else:
                self.run.style_simplify_inference(epoch, task_iter, task_idx=task_idx)

    def continual_eval(self, num_tasks=6, epoch=10):
        temp = self.run.train_config.result_path
        for i in range(num_tasks):
            self.run.train_config.result_path = temp + os.sep + self.task_names[orders[self.task_order][i]]
            logger.info(f"Task {self.task_names[orders[self.task_order][i]]} beginning...")
            self.run.eval(epoch)
