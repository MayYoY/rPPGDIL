import torch
import numpy as np

from . import postprocess


class Accumulate:
    def __init__(self, n, names=None):
        self.n = n
        self.cnt = [0] * n
        self.acc = [0] * n
        if names is None or len(names) != n:
            self.names = [f"Fold{i}" for i in range(n)]
        else:
            self.names = names

    def update(self, val: list, n):
        if not isinstance(n, list):
            n = [n] * self.n
        self.cnt = [a + b for a, b in zip(self.cnt, n)]
        self.acc = [a + b for a, b in zip(self.acc, val)]

    def reset(self):
        self.cnt = [0] * self.n
        self.acc = [0] * self.n

    def __repr__(self) -> str:
        info = "\n"
        for i in range(self.n):
            info += f"\t {self.names[i]}: {self.acc[i] / self.cnt[i]: .3f}\n"
        return info


def cal_metric(pred_phys: np.ndarray, label_phys: np.ndarray,
               methods=None) -> list:
    if methods is None:
        methods = ["Mean", "Std", "MAE", "RMSE", "MAPE", "R"]
    pred_phys = pred_phys.reshape(-1)
    label_phys = label_phys.reshape(-1)
    ret = [] * len(methods)
    for m in methods:
        if m == "Mean":
            ret.append((pred_phys - label_phys).mean())
        elif m == "Std":
            ret.append((pred_phys - label_phys).std())
        elif m == "MAE":
            ret.append(np.abs(pred_phys - label_phys).mean())
        elif m == "RMSE":
            ret.append(np.sqrt((np.square(pred_phys - label_phys)).mean()))
        elif m == "MAPE":
            ret.append((np.abs((pred_phys - label_phys) / label_phys)).mean() * 100)
        elif m == "R":
            temp = np.corrcoef(pred_phys, label_phys)
            if np.isnan(temp).any() or np.isinf(temp).any():
                ret.append(-1 * np.ones(1))
            else:
                ret.append(temp[0, 1])
    return ret
