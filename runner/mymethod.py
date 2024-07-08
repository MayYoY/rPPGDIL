from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from tools import distributed as mydist
from tools import logging
import math
from sklearn.cluster import KMeans
from . import Naive
from evaluate import metrics, postprocess
from timm.scheduler import CosineLRScheduler
import os


logger = logging.get_logger(__name__)


class MyMethod(Naive):
    def __init__(self, train_config, task_num=6, svd_bound=9, style2_dim=128, svd_dim=512, debug=False):
        super().__init__(train_config, debug)
        self.debug = debug
        self.task_num = task_num
        self.style_n_clusters = self.train_config.style_n_clusters
        self.noise_n_clusters = self.train_config.noise_n_clusters
        self.p = self.train_config.aug_p
        if hasattr(self.train_config, "svd_bound"):
            self.svd_bound = self.train_config.svd_bound
            print(f"SVD Bound: {self.svd_bound} !!!")
        else:
            self.svd_bound = svd_bound
        self.style2_dim = style2_dim
        self.svd_dim = svd_dim
        self.hard_style2_pool = []
        self.hard_noise_pool = []

        self.hard_style2_maes = [0.] * (self.style_n_clusters * (self.task_num - 1) + 1)
        self.hard_style2_cnt = [0] * (self.style_n_clusters * (self.task_num - 1) + 1)
        self.hard_noise_maes = [0.] * (self.style_n_clusters * (self.task_num - 1) + 1)
        self.hard_noise_cnt = [0] * (self.style_n_clusters * (self.task_num - 1) + 1)

        self.aug_cnt1, self.aug_cnt2, self.aug_cnt3 = 0, 0, 0
    
    def freeze_running_stat(self):
        for name, m in self.net.named_modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if "peft" not in name and "regressor" not in name:
                    m.eval()

    def before_task(self, data_iter, data_sampler, task_idx):
        if task_idx:
            self.num_epochs = self.train_config.contin_epoch
            local_lr = self.train_config.contin_lr
        else:
            self.net.backbone.freeze_peft("task0")
            logger.info("Freeze peft at task0")
            local_lr = self.train_config.init_lr
        self.data_iter = data_iter
        self.data_sampler = data_sampler
        
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=local_lr, 
                                          weight_decay=self.train_config.weight_decay, eps=1e-6)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=max(1, self.num_epochs), lr_min=1e-5, 
                                           warmup_t=self.train_config.warmup_t, warmup_lr_init=1e-5)
        self.net.train()
        if task_idx > 0:
            self.freeze_running_stat()
        
        ls = []
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                ls.append(name)
        logger.info(f"Learnable Params: {ls}")

    def train_task(self, task_idx):
        self.net.train()
        if task_idx > 0:
            self.freeze_running_stat()
        self.net.cuda()
        epoch_timer = logging.EpochTimer()
        for epoch in range(self.num_epochs):
            if self.data_sampler:
                print("Sampler exist!!!")
                self.data_sampler.set_epoch(epoch)
            self.scheduler.step(epoch)
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
                if epoch == self.num_epochs - 1:
                    os.makedirs(self.train_config.ckpt_path, exist_ok=True)
                    sd = self.net.module.state_dict() if self.train_config.num_gpus > 1 else self.net.state_dict()
                    torch.save(sd, self.train_config.ckpt_path + os.sep + f"{epoch + 1}.pt")
            mydist.synchronize()

    def post_task(self, task_idx):
        torch.cuda.empty_cache()
        if task_idx == self.task_num - 1:
            self.noise_n_clusters = 1
            self.style_n_clusters = 1
        if task_idx == 0:
            name = None
        else:
            name = "task0"
        logger.info("Updating pools...")
        bar = tqdm(range(len(self.data_iter)))
        self.net.eval()
        self.net.cuda()

        task_style2 = []
        task_noise = []

        for batch_data in self.data_iter:
            inputs = batch_data["input"].cuda()
            bs = len(inputs)
            
            with torch.no_grad():
                conv1_features = self.net.get_conv1_feature(inputs, name=name)
                conv2_features = self.net.conv1_to_conv2_feature(conv1_features, name=name)  # B x C x T x H x W

                regress_features = self.net.extract_feature(inputs, name=name, avg_pool=False)
                regress_features = regress_features.flatten(3).mean(-1)
            
            for idx in range(bs):
                mean2 = conv2_features[idx].flatten(1).mean(-1).detach()
                std2 = conv2_features[idx].flatten(1).std(-1).detach()
                task_style2.append(torch.cat([mean2, std2]))
                
                temp_feat = regress_features[idx].detach()
                u, s, vh = torch.linalg.svd(temp_feat, full_matrices=False)
                mask = torch.ones(temp_feat[1].shape, dtype=torch.float32).cuda()
                mask[:self.svd_bound] = 0.
                noise_feat = u @ (torch.diag(s) * mask) @ vh
                task_noise.append(noise_feat.view(self.svd_dim * 80))
            bar.update(1)
            if (not self.train_config.style_aug) and (not self.train_config.noise_aug):
                break
        
        temp = torch.stack(task_style2, 0).detach().cpu().numpy()

        clustering2 = KMeans(n_clusters=self.style_n_clusters, random_state=42).fit(temp)
        for i in range(self.style_n_clusters):
            self.hard_style2_pool.append(torch.tensor(clustering2.cluster_centers_[i]).cuda())

        task_noise = torch.stack(task_noise, 0).detach().cpu().numpy()
        clustering3 = KMeans(n_clusters=self.noise_n_clusters, random_state=42).fit(task_noise)
        for i in range(self.noise_n_clusters):
            self.hard_noise_pool.append(torch.tensor(clustering3.cluster_centers_[i]).cuda())

        logger.info(f"Current Size of Style2 Pool: {len(self.hard_style2_pool)}")
        logger.info(f"Current Size of Noise Pool: {len(self.hard_noise_pool)}")

        os.makedirs(self.train_config.ckpt_path, exist_ok=True)
        np.save(f"{self.train_config.ckpt_path}/task_style2.npy", torch.stack(task_style2).detach().cpu().numpy())
        np.save(f"{self.train_config.ckpt_path}/task_noise.npy", task_noise)

        np.save(f"{self.train_config.ckpt_path}/noise_pool.npy", torch.stack(self.hard_noise_pool).detach().cpu().numpy())
        np.save(f"{self.train_config.ckpt_path}/style2_pool.npy", torch.stack(self.hard_style2_pool).detach().cpu().numpy())
        if task_idx == self.task_num - 1:
            np.save(f"{self.train_config.ckpt_path}/style2_maes.npy", np.asarray(self.hard_style2_maes))
            np.save(f"{self.train_config.ckpt_path}/noise_maes.npy", np.asarray(self.hard_noise_maes))

        self.net.train()
        if task_idx > 0:
            self.freeze_running_stat()
        if task_idx == 0:
            self.optimizer.zero_grad()
            self.optimizer = None
            self.net.backbone.freeze_model()
            self.net.dn.freeze_model()
            logger.info("No Freeze Regressor!!!")
            self.net.backbone.unfreeze_peft("task0")

    def train_epoch(self, epoch, task_idx=None):
        if task_idx == 0:
            name = None
        else:
            name = "task0"
        for batch_data in self.data_iter:
            x = batch_data["input"].cuda()
            y = batch_data["wave"].cuda()
            average_hr = batch_data["wave_hr"].view(-1).cuda()
            fs = batch_data["fs"].view(-1).cuda()
            bs = x.shape[0]

            aug_means2, aug_stds2, aug_style2_indices = self.random_style2_indices(bs)
            aug_noise, aug_noise_indices = self.random_noise(bs)

            conv1_features = self.net.get_conv1_feature(x, name=name)
            conv2_features = self.net.conv1_to_conv2_feature(conv1_features, name=name)
            if aug_means2 is not None:
                means2 = conv2_features.flatten(2).mean(-1)
                stds2 = conv2_features.flatten(2).std(-1)
                conv2_features = (conv2_features - means2[:, :, None, None, None]) / (stds2[:, :, None, None, None] + 1e-6)
                conv2_features = conv2_features * aug_stds2[:, :, None, None, None] + aug_means2[:, :, None, None, None]
            regress_features = self.net.conv_to_regress(conv2_features, name=name, avg_pool=False)
            regress_features = regress_features.flatten(3).mean(-1)  # B C T
            if aug_noise is not None:
                u, s, vh = torch.linalg.svd(regress_features, full_matrices=False)
                batch_singular = []
                for i in range(len(s)):
                    batch_singular.append(torch.diag(s[i]))
                batch_singular = torch.stack(batch_singular)
                mask = torch.ones(regress_features.shape[2], dtype=torch.float32).cuda()
                mask[self.svd_bound:] = 0.
                regress_features = u @ (batch_singular * mask[None, :]) @ vh + aug_noise
            preds = self.net.forward_regress_feature(regress_features, name=name)

            rppg_loss = self.loss_fun1(preds, y)
            dist_loss, freq_loss, mae_loss, mae_per_sample = self.loss_fun2(preds, average_hr, fs, raw_mae=True)
            a = 0.1 * math.pow(0.5, epoch / self.num_epochs)
            b = math.pow(5., epoch / self.num_epochs)
            loss = a * rppg_loss + b * (freq_loss + dist_loss)
            if loss.isnan().any():
                raise ValueError("NaN comes from total loss")

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            self.post_iter()
            
            if task_idx == self.task_num - 1 and aug_means2 is not None:
                for i in range(bs):
                    self.hard_style2_cnt[aug_style2_indices[i]] += 1
                    temp = self.hard_style2_cnt[aug_style2_indices[i]]
                    self.hard_style2_maes[aug_style2_indices[i]] = float(mae_per_sample[i]) / temp + self.hard_style2_maes[aug_style2_indices[i]] * (temp - 1) / temp
            elif task_idx == self.task_num - 1:
                for i in range(bs):
                    self.hard_style2_cnt[-1] += 1
                    temp = self.hard_style2_cnt[-1]
                    self.hard_style2_maes[-1] = float(mae_per_sample[i]) / temp + self.hard_style2_maes[-1] * (temp - 1) / temp
            if task_idx == self.task_num - 1 and aug_noise is not None:
                for i in range(bs):
                    self.hard_noise_cnt[aug_noise_indices[i]] += 1
                    temp = self.hard_noise_cnt[aug_noise_indices[i]]
                    self.hard_noise_maes[aug_noise_indices[i]] = float(mae_per_sample[i]) / temp + self.hard_noise_maes[aug_noise_indices[i]] * (temp - 1) / temp
            elif task_idx == self.task_num - 1:
                for i in range(bs):
                    self.hard_noise_cnt[-1] += 1
                    temp = self.hard_noise_cnt[-1]
                    self.hard_noise_maes[-1] = float(mae_per_sample[i]) / temp + self.hard_noise_maes[-1] * (temp - 1) / temp

            if self.train_config.num_gpus > 1:
                [rppg_loss, freq_loss, dist_loss, mae_loss] = mydist.all_reduce([rppg_loss, freq_loss, dist_loss, mae_loss])
            val = [rppg_loss.data, freq_loss.data, dist_loss.data, mae_loss.data]
            self.train_loss.update(val=val, n=1)

    def random_style2_indices(self, bs):
        """ Conv2 Style """
        if not self.train_config.style_aug or torch.rand(1) < self.p or len(self.hard_style2_pool) == 0:
            return None, None, None
        means = []
        indices = []
        stds = []
        for _ in range(bs):
            idx = random.randint(0, len(self.hard_style2_pool) - 1)
            indices.append(idx)
            means.append(self.hard_style2_pool[idx][:self.style2_dim])
            stds.append(self.hard_style2_pool[idx][self.style2_dim:])
        if self.aug_cnt2 == 0:
            logger.info("Random Style2 Works!")
            self.aug_cnt2 += 1
        return torch.stack(means).cuda(), torch.stack(stds).cuda(), indices
    
    def random_noise(self, bs):
        if not self.train_config.noise_aug or torch.rand(1) < self.p or len(self.hard_noise_pool) == 0:
            return None, None
        noise_regress_feats = []
        indices = []
        for _ in range(bs):
            idx = random.randint(0, len(self.hard_noise_pool) - 1)
            noise_regress_feats.append(self.hard_noise_pool[idx])
            indices.append(idx)
        if self.aug_cnt3 == 0:
            logger.info("Random SVD Noise Works!")
            self.aug_cnt3 += 1
        return torch.stack(noise_regress_feats).reshape(bs, self.svd_dim, 80).cuda(), indices  # B x C x T

    def style_simplify_inference(self, epoch, test_iter, task_idx=0):
        style2_maes = np.load(f"{self.train_config.ckpt_path}/style2_maes.npy")
        best_idx = np.argmin(style2_maes)
        style2_mean = np.load(f"{self.train_config.ckpt_path}/style2_pool.npy")[best_idx, :self.style2_dim]
        raw_aug_means2 = torch.from_numpy(style2_mean).view(1, -1).cuda()
        style2_std = np.load(f"{self.train_config.ckpt_path}/style2_pool.npy")[best_idx, self.style2_dim:]
        raw_aug_stds2 = torch.from_numpy(style2_std).view(1, -1).cuda()

        self.net.load_state_dict(torch.load(f"{self.train_config.ckpt_path}/{epoch}.pt", map_location="cpu"), strict=True)
        self.net.cuda()
        self.net.eval()
        predictions_adapter = []
        gt_hrs = []
        wave_hrs = []
        frame_rates = []
        indices = dict()
        subject_ls = {"subject": []}
        bar = tqdm(range(len(test_iter)))
        for batch_data in test_iter:
            x = batch_data["input"].to(self.train_config.gpu_id)
            gt_hr = batch_data["gt_hr"].view(-1)
            wave_hr = batch_data["wave_hr"].view(-1)
            fs = batch_data["fs"].view(-1)
            subjects = batch_data["subject"]

            aug_means2 = raw_aug_means2.repeat(len(fs), 1)
            aug_stds2 = raw_aug_stds2.repeat(len(fs), 1)
            with torch.no_grad():
                conv1_features = self.net.get_conv1_feature(x, name="task0")
                conv2_features = self.net.conv1_to_conv2_feature(conv1_features, name="task0")
                means2 = conv2_features.flatten(2).mean(-1)
                stds2 = conv2_features.flatten(2).std(-1)  # NOTE: AdaIN
                conv2_features = (conv2_features - means2[:, :, None, None, None]) / (stds2[:, :, None, None, None] + 1e-6)
                conv2_features = conv2_features * aug_stds2[:, :, None, None, None] + aug_means2[:, :, None, None, None]
                regress_features = self.net.conv_to_regress(conv2_features, name="task0", avg_pool=False)
                regress_features = regress_features.flatten(3).mean(-1)  # B C T
                output_adapter = self.net.forward_regress_feature(regress_features, name="task0")

            for i in range(len(x)):
                file_name = subjects[i]
                if file_name not in indices.keys():
                    indices[file_name] = len(predictions_adapter)
                    predictions_adapter.append([])
                    gt_hrs.append([])
                    wave_hrs.append([])
                    frame_rates.append(float(fs[i]))
                    subject_ls["subject"].append(file_name)
                predictions_adapter[indices[file_name]].append(output_adapter[i].detach().cpu().numpy())
                gt_hrs[indices[file_name]].append(gt_hr[i].detach().cpu().numpy())
                wave_hrs[indices[file_name]].append(wave_hr[i].detach().cpu().numpy())
            bar.update(1)
        os.makedirs(f"{self.train_config.result_path}/{epoch}", exist_ok=True)
        np.savez(f"{self.train_config.result_path}/{epoch}/predictions_style_simplify.npz", *predictions_adapter)
        np.savez(f"{self.train_config.result_path}/{epoch}/gt_hrs.npz", *gt_hrs)
        np.savez(f"{self.train_config.result_path}/{epoch}/wave_hrs.npz", *wave_hrs)
        np.save(f"{self.train_config.result_path}/{epoch}/frame_rates.npy", frame_rates, allow_pickle=True)
        subject_ls = pd.DataFrame(subject_ls)
        subject_ls.to_csv(f"{self.train_config.result_path}/{epoch}/subject_list.csv", index=False)

    def inference(self, epoch, test_iter, task_idx=0):
        self.net.load_state_dict(torch.load(f"{self.train_config.ckpt_path}/{epoch}.pt", map_location="cpu"), strict=True)
        self.net.cuda()

        self.net.eval()
        predictions_adapter = []
        gt_hrs = []
        wave_hrs = []
        frame_rates = []
        indices = dict()
        bar = tqdm(range(len(test_iter)))
        for batch_data in test_iter:
            x = batch_data["input"].to(self.train_config.gpu_id)
            gt_hr = batch_data["gt_hr"].view(-1)
            wave_hr = batch_data["wave_hr"].view(-1)
            fs = batch_data["fs"].view(-1)
            subjects = batch_data["subject"]
            with torch.no_grad():
                output_adapter = self.net(x, name="task0")

            for i in range(len(x)):
                file_name = subjects[i]
                if file_name not in indices.keys():
                    indices[file_name] = len(predictions_adapter)
                    predictions_adapter.append([])
                    gt_hrs.append([])
                    wave_hrs.append([])
                    frame_rates.append(float(fs[i]))
                predictions_adapter[indices[file_name]].append(output_adapter[i].detach().cpu().numpy())
                gt_hrs[indices[file_name]].append(gt_hr[i].detach().cpu().numpy())
                wave_hrs[indices[file_name]].append(wave_hr[i].detach().cpu().numpy())
            bar.update(1)
        os.makedirs(f"{self.train_config.result_path}/{epoch}", exist_ok=True)
        np.savez(f"{self.train_config.result_path}/{epoch}/predictions_adapter.npz", *predictions_adapter)
        np.savez(f"{self.train_config.result_path}/{epoch}/gt_hrs.npz", *gt_hrs)
        np.savez(f"{self.train_config.result_path}/{epoch}/wave_hrs.npz", *wave_hrs)
        np.save(f"{self.train_config.result_path}/{epoch}/frame_rates.npy", frame_rates, allow_pickle=True)

    def eval(self, epoch):
        logger.info(self.train_config.result_path)
        # predictions_adapter = np.load(f"{self.train_config.result_path}/{epoch}/predictions_adapter.npz")
        predictions_adapter = np.load(f"{self.train_config.result_path}/{epoch}/predictions_style_simplify.npz")
        predictions_adapter = [predictions_adapter[k] for k in predictions_adapter]

        labels = np.load(f"{self.train_config.result_path}/{epoch}/gt_hrs.npz")
        labels = [labels[k] for k in labels]
        frame_rates = np.load(f"{self.train_config.result_path}/{epoch}/frame_rates.npy")

        pred_adapter_phys = []
        label_phys = []
        bar = tqdm(range(len(predictions_adapter)))
        for i in range(len(predictions_adapter)):
            pred_adapter_temp = postprocess.fft_physiology(predictions_adapter[i], Fs=frame_rates[i], 
                                                           diff=False, detrend_flag=True).reshape(-1)

            pred_adapter_phys.append(pred_adapter_temp.mean())
            label_phys.append(labels[i].mean())
            bar.update(1)
        pred_adapter_phys = np.asarray(pred_adapter_phys)
        label_phys = np.asarray(label_phys)

        results_adapter = metrics.cal_metric(pred_adapter_phys, label_phys)  # "Mean", "Std", "MAE", "RMSE", "MAPE", "R"
        logger.info(
            f"========= Epoch {epoch} Adapter Results =========\n"
            f"\t Mean: {results_adapter[0]: .3f}\n"
            f"\t Std: {results_adapter[1]: .3f}\n"
            f"\t MAE: {results_adapter[2]: .3f}\n"
            f"\t RMSE: {results_adapter[3]: .3f}\n"
            f"\t MAPE: {results_adapter[4]: .3f}\n"
            f"\t R: {results_adapter[5]: .3f}\n"
        )
