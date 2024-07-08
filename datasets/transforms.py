import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv


def interpolate_data(frames, wave, target_len):
    _, _, H, W = frames.shape
    ret_frames = F.interpolate(frames.unsqueeze(0), size=(target_len, H, W), mode="trilinear")[0]
    ret_wave = F.interpolate(wave.unsqueeze(0).unsqueeze(0), size=(target_len,), mode="linear")
    return ret_frames, ret_wave[0, 0]


def read_video(video_path, start, end):
    frames = []
    for i in range(start, end):
        img = cv.imread(f"{video_path}/{i}.png")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # H W C
        frames.append(img)
    frames = torch.FloatTensor(np.asarray(frames))  # T H W C
    frames = frames.permute(3, 0, 1, 2)  # C T H W

    return frames


def resample_from_path(video_path, wave, hr, start, end, video_len, p=0.5):
    if torch.rand(1) < p:
        frames = read_video(video_path, start, end)
        assert frames.shape[1] == 160
        return frames, wave[start: end], 1  # speed
    clip_len = end - start

    if 60 <= hr <= 110:
        # downsample 0.67, hr * 1.5
        new_end = round(start + 1.5 * clip_len)
        if new_end > video_len:  # NOTE: end
            frames = read_video(video_path, start, end)
            assert frames.shape[1] == 160
            return frames, wave[start: end], 1
        # NOTE: new_end
        frames = read_video(video_path, start, new_end)
        frames, wave = interpolate_data(frames, wave[start: new_end], target_len=clip_len)
        return frames, wave, 1.5
    elif 70 <= hr <= 85:
        # upsample 1.5, hr * 0.67
        new_end = round(start + 0.67 * clip_len)
        frames = read_video(video_path, start, new_end)
        frames, wave = interpolate_data(frames, wave[start: new_end], target_len=clip_len)
        return frames, wave, 0.67
    else:
        frames = read_video(video_path, start, end)
        return frames, wave[start: end], 1


class RandomGaussianNoise(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, clip):
        if torch.rand(1) < self.p:
            return clip + torch.normal(0., 2., size=clip.shape)
        return clip


class RandomIlluminationNoise(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, clip):
        if torch.rand(1) < self.p:
            return clip + torch.normal(0., 10., (1,))
        return clip


class RandomResizedCrop3D(nn.Module):
    def __init__(self, scale=(0.5, 1.)):
        super().__init__()
        self.scale = scale

    def forward(self, clip):
        _, T, H, _ = clip.shape
        crop_scale = np.random.uniform(self.scale[0], self.scale[1])
        crop_length = np.round(crop_scale * H).astype(int)
        crop_start_lim = H - crop_length
        x1 = np.random.randint(0, crop_start_lim + 1)
        # y1 = x1
        x2 = x1 + crop_length
        # y2 = y1 + crop_length
        # cropped_clip = clip[:, :, y1:y2, x1:x2]
        cropped_clip = clip[:, :, x1:x2, x1:x2]
        resized_clip = F.interpolate(cropped_clip.unsqueeze(0), (T, H, H), mode='trilinear', align_corners=False)[0]

        return resized_clip
