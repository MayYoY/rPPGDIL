import torch
import argparse
import subprocess
from scripts import clapp_new
from configs import contintrain, contintest


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    app = clapp_new.App(contintrain.MyMethodCfg(), args, "train", 0)
    app.continual_train()
    # app = clapp_new.App(contintest.MyMethodCfg(), args, "test", 0)
    # app.continual_inference()
    # app.continual_eval(6)
