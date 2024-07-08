import numpy as np
import pandas as pd
import cv2 as cv
from mtcnn import MTCNN
from math import ceil
import scipy
import scipy.io
from scipy.signal import butter, periodogram
from scipy.sparse import spdiags
import os
import csv
import copy
from scipy import interpolate
import math


def resize(frames, dynamic_det, det_length,
           w, h, larger_box, crop_face, larger_box_size):
    if dynamic_det:
        det_num = ceil(len(frames) / det_length)
    else:
        det_num = 1
    face_region = []
    detector = MTCNN()
    # detector = None
    for idx in range(det_num):
        if crop_face:
            face_region.append(facial_detection(detector, frames[det_length * idx],
                                                larger_box, larger_box_size))
        else:
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    resize_frames = []

    for i in range(len(frames)):
        frame = frames[i]
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[3], frame.shape[0]),  # h
                          max(face_region[0], 0):min(face_region[2], frame.shape[1])]  # w
        if w > 0 and h > 0:
            resize_frames.append(cv.resize(frame, (w + 4, h + 4),
                                           interpolation=cv.INTER_CUBIC)[2: w + 2, 2: h + 2, :])
        else:
            resize_frames.append(frame)
    if w > 0 and h > 0:
        return np.asarray(resize_frames)
    else:  # list
        return resize_frames


def facial_detection(detector, frame, larger_box=False, larger_box_size=1.0):
    face_zone = detector.detect_faces(frame)
    if len(face_zone) < 1:
        print("Warning: No Face Detected!")
        return [0, 0, frame.shape[0], frame.shape[1]]
    if len(face_zone) >= 2:
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    result = face_zone[0]['box']
    result[2] += result[0]
    result[3] += result[1]
    return result


def chunk(x, chunk_length, chunk_stride=-1):
    if chunk_stride < 0:
        chunk_stride = chunk_length
    x_clips = [x[i: i + chunk_length] for i in range(0, x.shape[0] - chunk_length + 1, chunk_stride)]
    return np.array(x_clips)


def normalize_frame(frame):
    return (frame - 127.5) / 128


def standardize(data):
    data = data - np.mean(data)
    data = data / (np.std(data) + 1e-6)
    data[np.isnan(data)] = 0
    return data


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(signal, lambda_value=100):
    T = signal.shape[-1]
    # observation matrix
    H = np.identity(T)  # T x T
    ones = np.ones(T)  # T,
    minus_twos = -2 * np.ones(T)  # T,
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (T - 2), T).toarray()
    designal = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T.dot(D))).dot(signal.T).T
    return designal


def fft_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    # get filter and detrend
    if target == "pulse":
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # get psd
    N = next_power_of_2(signal.shape[-1])
    freq, psd = periodogram(signal, fs=Fs, nfft=N, detrend=False)
    # get mask
    if target == "pulse":
        mask = np.argwhere((freq >= 0.75) & (freq <= 2.5))
    else:
        mask = np.argwhere((freq >= 0.08) & (freq <= 0.5))
    # get peak
    freq = freq[mask]
    if len(signal.shape) == 1:
        idx = psd[mask.reshape(-1)].argmax(-1)
    else:
        idx = psd[:, mask.reshape(-1)].argmax(-1)
    phys = freq[idx] * 60
    return phys.reshape(-1)
