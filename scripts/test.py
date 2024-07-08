import torch
import torch.nn as nn
import pprint
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from models import build
from tools import logging
from datasets import rppgdata
from configs import contintest
from evaluate import metrics, postprocess


logger = logging.get_logger(__name__)


def inference(test_config, epoch, verbal=True, test_iter=None, debug=False, net=None):
    if debug:
        test_config.gpu_id = "cpu"
        test_config.num_gpus = 0
    print(f"Epoch: {epoch}")
    if verbal:
        print("Inference with config:")
        print(pprint.pformat(test_config))
    torch.cuda.set_device(test_config.gpu_id)
    if test_iter is None:
        test_set = rppgdata.VIPLDataset(test_config)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size=test_config.batch_size,
                                                shuffle=False, num_workers=8)
        print(f"Length of test set {len(test_set)}, test iter {len(test_iter)}")
    else:
        print(f"Length of test iter {len(test_iter)}")
    if net is None:
        net = build.build_model(test_config)
    net.load_state_dict(torch.load(f"{test_config.ckpt_path}/{epoch}.pt", map_location="cpu"), strict=True)
    net = net.to(test_config.gpu_id)

    net.eval()
    predictions = []
    gt_hrs = []
    wave_hrs = []
    frame_rates = []
    indices = dict()
    subject_ls = {"subject": []}
    bar = tqdm(range(len(test_iter)))
    for batch_data in test_iter:
        with torch.no_grad():
            x = batch_data["input"].to(test_config.gpu_id)
            gt_hr = batch_data["gt_hr"].view(-1).to(test_config.gpu_id)
            wave_hr = batch_data["wave_hr"].view(-1).to(test_config.gpu_id)
            fs = batch_data["fs"].view(-1).to(test_config.gpu_id)
            subjects = batch_data["subject"]
            preds = net(x, name="task0")

        for i in range(len(x)):
            file_name = subjects[i]
            if file_name not in indices.keys():
                indices[file_name] = len(predictions)
                predictions.append([])
                gt_hrs.append([])
                wave_hrs.append([])
                frame_rates.append(float(fs[i]))
                subject_ls["subject"].append(file_name)
            predictions[indices[file_name]].append(preds[i].detach().cpu().numpy())
            gt_hrs[indices[file_name]].append(gt_hr[i].detach().cpu().numpy())
            wave_hrs[indices[file_name]].append(wave_hr[i].detach().cpu().numpy())
        bar.update(1)
    os.makedirs(f"{test_config.result_path}/{epoch}", exist_ok=True)
    np.savez(f"{test_config.result_path}/{epoch}/predictions.npz", *predictions)
    np.savez(f"{test_config.result_path}/{epoch}/gt_hrs.npz", *gt_hrs)
    np.savez(f"{test_config.result_path}/{epoch}/wave_hrs.npz", *wave_hrs)
    np.save(f"{test_config.result_path}/{epoch}/frame_rates.npy", frame_rates, allow_pickle=True)
    subject_ls = pd.DataFrame(subject_ls)
    subject_ls.to_csv(f"{test_config.result_path}/{epoch}/subject_list.csv", index=False)


def eval(test_config, epoch, verbal=True):
    logging.setup_logging(test_config.log_path)
    logger.info(f"Epoch: {epoch}")
    if verbal:
        logger.info(pprint.pformat(test_config))
    predictions = np.load(f"{test_config.result_path}/{epoch}/predictions.npz")
    predictions = [predictions[k] for k in predictions]
    labels = np.load(f"{test_config.result_path}/{epoch}/gt_hrs.npz")
    labels = [labels[k] for k in labels]
    frame_rates = np.load(f"{test_config.result_path}/{epoch}/frame_rates.npy")

    pred_phys = []
    label_phys = []
    bar = tqdm(range(len(predictions)))
    for i in range(len(predictions)):
        pred_temp = postprocess.fft_physiology(predictions[i], Fs=frame_rates[i],
                                               diff=False, detrend_flag=True).reshape(-1)
        pred_phys.append(pred_temp.mean())
        label_phys.append(labels[i].mean())
        bar.update(1)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    results = metrics.cal_metric(pred_phys, label_phys)  # "Mean", "Std", "MAE", "RMSE", "MAPE", "R"
    logger.info(
        f"========= Epoch {epoch} =========\n"
        f"\t Mean: {results[0]: .3f}\n"
        f"\t Std: {results[1]: .3f}\n"
        f"\t MAE: {results[2]: .3f}\n"
        f"\t RMSE: {results[3]: .3f}\n"
        f"\t MAPE: {results[4]: .3f}\n"
        f"\t R: {results[5]: .3f}\n"
    )
