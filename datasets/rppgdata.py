import torch
from torch.utils import data
import torchvision
import pandas as pd
import numpy as np
import collections
from . import transforms


class RPPGDataset(data.Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.data = self.get_data()
        self.transforms = self.get_transforms()

    def get_data(self):
        ret = collections.defaultdict(list)
        record = pd.read_csv(self.config.record)
        for i in range(len(record)):
            if self.isvalid(record, i):
                ret["input_files"].append(record.loc[i, "input_files"])
                ret["wave_files"].append(record.loc[i, "wave_files"])
                ret["start"].append(record.loc[i, "start"])
                ret["end"].append(record.loc[i, "end"])
                ret["video_len"].append(record.loc[i, "video_len"])
                ret["gt_HR"].append(record.loc[i, "gt_HR"])
                ret["wave_HR"].append(record.loc[i, "wave_HR"])
                if "Fs" in record.columns:
                    ret["Fs"].append(record.loc[i, "Fs"])
                else:
                    ret["Fs"].append(30.)  # default fps
        return ret

    def get_transforms(self):
        ret = []
        if "filp" in self.config.transforms:
            ret.append(torchvision.transforms.RandomHorizontalFlip())
        if "illusion" in self.config.transforms:
            ret.append(transforms.RandomIlluminationNoise())
        if "gauss" in self.config.transforms:
            ret.append(transforms.RandomGaussianNoise())
        if "crop" in self.config.transforms:
            ret.append(transforms.RandomResizedCrop3D())
        ret.append(torchvision.transforms.Resize(self.config.input_resolution))
        return torchvision.transforms.Compose(ret)

    def isvalid(self):
        NotImplemented

    def get_subject(self, idx):
        item_path = self.data["wave_files"][idx]
        subject_id = item_path.split('/')[-1]
        clip_len = self.data["end"][idx] - self.data["start"][idx]
        chunk_idx = self.data["start"][idx] / clip_len
        return subject_id

    def __len__(self):
        return len(self.data["input_files"])

    def __getitem__(self, idx):
        left, right = self.data["start"][idx], self.data["end"][idx]
        video_len = self.data["video_len"][idx]
        video_path = self.data['input_files'][idx]
        wave = torch.from_numpy(np.load(self.data["wave_files"][idx]))
        gt_hr = torch.FloatTensor([self.data["gt_HR"][idx]])
        wave_hr = torch.FloatTensor([self.data["wave_HR"][idx]])
        fs = torch.FloatTensor([self.data["Fs"][idx]])

        if "resample" in self.config.transforms:
            frames, wave, speed = transforms.resample_from_path(video_path, wave, wave_hr, left, right, video_len)
        else:
            frames = transforms.read_video(video_path, left, right)  # C T H W
            wave = wave[left: right]
            speed = 1

        frames = self.transforms(frames)
        frames = (torch.clip(frames, 0., 255.) - 127.5) / 128

        return {"input": frames, "wave": wave, "gt_hr": gt_hr * speed,
                "wave_hr": wave_hr * speed, "fs": fs, "subject": self.get_subject(idx),
                "start": self.data["start"][idx]}


class VIPLDataset(RPPGDataset):
    def __init__(self, config):
        super().__init__(config)

    def isvalid(self, record, i):
        if record.loc[i, "fold"] not in self.config.folds:
            return False
        if self.config.task and record.loc[i, "task"] not in self.config.task:
            return False
        if self.config.source and record.loc[i, "source"] not in self.config.source:
            return False
        return True


class UPDataset(RPPGDataset):
    """ UBFC PURE """

    def __init__(self, config):
        super().__init__(config)

    def isvalid(self, record, i):
        return (record.loc[i, "split"]).lower() == self.config.split.lower()


class BUAADataset(RPPGDataset):
    """ BUAA """

    def __init__(self, config):
        super().__init__(config)

    def isvalid(self, record, i):
        return (record.loc[i, "split"]).lower() == self.config.split.lower() and record.loc[i, "lux"] >= self.config.lux


class MMPDDataset(RPPGDataset):
    """ MMPD """

    def __init__(self, config):
        super().__init__(config)

    def isvalid(self, record, i):
        if self.config.light and record.loc[i, "light"] not in self.config.light:
            return False
        if self.config.motion and record.loc[i, "motion"] not in self.config.motion:
            return False
        if self.config.exercise and record.loc[i, "exercise"] not in self.config.exercise:
            return False
        if self.config.skin_color and record.loc[i, "skin_color"] not in self.config.skin_color:
            return False
        if self.config.gender and record.loc[i, "gender"] not in self.config.gender:
            return False
        if self.config.glasser and record.loc[i, "glasser"] not in self.config.glasser:
            return False
        if self.config.hair_cover and record.loc[i, "hair_cover"] not in self.config.hair_cover:
            return False
        if self.config.makeup and record.loc[i, "makeup"] not in self.config.makeup:
            return False

        return (record.loc[i, "split"]).lower() == self.config.split.lower()
