import cv2 as cv
import numpy as np
import pandas as pd
import os
import glob
import scipy.io as scio
from tqdm.auto import tqdm
from scipy import interpolate, io

from . import utils


class FramePreprocess:
    def __init__(self, config):
        self.config = config
        self.dirs = glob.glob(self.config.input_path + os.sep + "*")

    @staticmethod
    def get_fold(subid):
        idx = int(subid)
        if idx <= 9:
            return "train"
        else:
            return "test"

    def read_process(self):
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        csv_info = {"input_files": [], "wave_files": [], "start": [], "end": [], "video_len": [], 
                    "split": [], "gt_HR": [], "wave_HR": [], "Fs": [], "lux": []}
        for subdir in self.dirs:  # i_th subject
            subid = subdir.split(os.sep)[-1][4:]  # Sub 01
            luxdirs = glob.glob(subdir + os.sep + "*")
            for luxdir in luxdirs:
                luxid = luxdir.split(os.sep)[-1][4:]  # lux 1.0
                filename = f"sub{subid}_lux{luxid}"

                clip_range, Fs, end_time = self.read_video(luxdir, filename)  # T,
                try:
                    csv_paths = glob.glob(luxdir + os.sep + "*.csv")
                    hr_path = csv_paths[1] if "wave" in csv_paths[0] else csv_paths[0]
                except:
                    print(luxdir)
                    raise ValueError("Could not find ground truth files")
                hrs = self.read_hrs(hr_path)

                if end_time <= len(hrs):
                    hrs = hrs[: end_time]
                else:
                    clip_range = clip_range[: round(len(hrs) * Fs)]
                if len(clip_range) < self.config.CHUNK_LENGTH:
                    continue

                waves = self.read_wave(luxdir)  # T_w,
                fun = interpolate.CubicSpline(range(len(waves)), waves)
                x_new = np.linspace(0, len(waves) - 1, num=len(clip_range))
                gts = fun(x_new)  # T

                # n x len, T
                frames_clips, standardized_gts = self.preprocess(clip_range, gts)
                single_info = {"filename": filename, "split": self.get_fold(subid), "Fs": Fs}
                temp = self.save(frames_clips, standardized_gts, hrs, single_info, Fs)
                csv_info["start"] += temp[0]
                csv_info["end"] += temp[1]
                csv_info["wave_HR"] += temp[2]
                csv_info["gt_HR"] += temp[3]
                N = len(frames_clips)
                csv_info["input_files"] += [self.config.img_cache + os.sep + f"{single_info['filename']}"] * N
                csv_info["wave_files"] += [self.config.gt_cache + os.sep + f"{single_info['filename']}.npy"] * N
                csv_info["video_len"] += [len(standardized_gts)] * N
                csv_info["split"] += [single_info["split"]] * N
                csv_info["Fs"] += [single_info["Fs"]] * N
                csv_info["lux"] += [float(luxid)] * N
            progress_bar.update(1)

        csv_info = pd.DataFrame(csv_info)
        csv_info.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, standardized_gts: np.array,
             hrs: np.ndarray, single_info: dict, Fs: float):
        start_list = []
        end_list = []
        wave_HR_list = []
        gt_HR_list = []
        for i in range(len(frames_clips)):
            left = frames_clips[i, 0]
            right = frames_clips[i, -1] + 1
            gt_HR_list.append(hrs[round(left / Fs): round(right / Fs)].mean())
            wave_HR = utils.fft_physiology(signal=standardized_gts[left: right], Fs=Fs, diff=False, detrend_flag=True)
            wave_HR_list.append(float(wave_HR))
            start_list.append(left)
            end_list.append(right)
        if self.config.MODIFY:
            os.makedirs(self.config.gt_cache, exist_ok=True)
            label_path = self.config.gt_cache + os.sep + f"{single_info['filename']}.npy"
            np.save(label_path, standardized_gts)
        return start_list, end_list, wave_HR_list, gt_HR_list

    def preprocess(self, clip_range, gts):
        standardized_gts = utils.standardize(gts[:])
        if self.config.DO_CHUNK:
            frames_clips = utils.chunk(clip_range, self.config.CHUNK_LENGTH, self.config.CHUNK_STRIDE)
        else:
            frames_clips = np.array([clip_range])  # n x len x H x W x C

        return frames_clips, standardized_gts

    def read_video(self, data_path, filename):
        save_dir = self.config.img_cache + os.sep + filename
        if self.config.MODIFY:
            try:
                video_path = glob.glob(data_path + os.sep + "*.avi")[0]
            except:
                print(data_path)
                raise ValueError("Could not find video file")
            vid = cv.VideoCapture(video_path)
            vid.set(cv.CAP_PROP_POS_MSEC, 0)
            ret, frame = vid.read()
            frames = list()
            while ret:
                frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)  # NOTE
                frame = np.asarray(frame)
                frame[np.isnan(frame)] = 0
                frames.append(frame)
                ret, frame = vid.read()
            vid.release()  # !!!
            frames = np.asarray(frames)

            if self.config.METHOD == "MTCNN":
                frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                                    self.config.DYNAMIC_DETECTION_FREQUENCY,
                                    self.config.W, self.config.H,
                                    self.config.LARGE_FACE_BOX,
                                    self.config.CROP_FACE,
                                    self.config.LARGE_BOX_COEF).astype(np.uint8)
            else:
                raise ValueError
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                cv.imwrite(save_dir + os.sep + f"{i}.png", frame)
            T = len(frames)
        else:
            T = len(glob.glob(save_dir + os.sep + "*.png"))
        # for return
        clips_range = np.arange(T)
        Fs = 30.
        end_time = round(T / Fs)
        
        return clips_range, Fs, end_time

    @staticmethod
    def read_wave(data_path):
        try:
            waves = scio.loadmat(data_path + os.sep + "PPGData.mat")['PPG']['data'][0][0].reshape(-1)
        except:
            wave_path = glob.glob(data_path + os.sep + "*wave.csv")[0]
            waves = pd.read_csv(wave_path, header=None).iloc[:, 0].values.reshape(-1)
        return waves

    @staticmethod
    def read_hrs(hr_path):
        hrs = pd.read_csv(hr_path)["PULSE"].values
        return hrs
