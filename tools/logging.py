#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson
import numpy as np
from fvcore.common.timer import Timer
from iopath.common.file_io import g_pathmgr
from . import distributed as du


def _suppress_print():
    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = g_pathmgr.open(
        filename, "a", buffering=1024 if "://" in filename else -1
    )
    atexit.register(io.close)
    return io


def setup_logging(output_file="./debug.log"):
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc(du.get_world_size()):  # du.is_master_proc(du.get_world_size())  du.is_root_proc()
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc(du.get_world_size()):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if du.is_master_proc(du.get_world_size()):
        fh = logging.StreamHandler(_cached_log_stream(output_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    return logging.getLogger(name)


class EpochTimer:

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        self.timer.reset()

    def epoch_toc(self):
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)

    def sum_epoch_time(self):
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.sum(self.epoch_times)
