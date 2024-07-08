from . import Cfg


class MyMethodCfg(Cfg):
    continual_pretrain = None
    pretrain = False
    freeze = False
    diffnorm = True
    peft = "adapter"

    style_n_clusters = 8
    style_aug = False
    noise_n_clusters = 8
    noise_aug = False
    aug_p = 0.5

    init_epoch = 20
    contin_epoch = 10
    log_path = "./logs/cl_mymethod_test.log"
    result_path = ""
    ckpt_path = ""
    batch_size = 4

    gpu_id = 0
    model_name = "uniformer_small"
    input_resolution = 96
    num_frames = 160
    num_gpus = 1

    name = "My method"


class ViplF14(Cfg):
    record = "mtcnn_vipl_s80_test.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    folds = [1, 2, 3, 4]
    task = None
    source = [1, 2, 3]


class ViplF5(Cfg):
    record = "mtcnn_vipl_s80_test.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    folds = [5]
    task = None
    source = [1, 2, 3]


class Pure(Cfg):
    record = "mtcnn_pure_s80.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    split = "test"


class Ubfc(Cfg):
    record = "mtcnn_ubfc_s80.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    split = "test"


class Buaa(Cfg):
    record = "mtcnn_buaa_s80.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    split = "test"
    lux = 6.3


class Mmpd_new(Cfg):
    record = "mtcnn_mmpd_new_s80.csv"
    transforms = []
    input_resolution = 96
    num_frames = 160
    split = "test"

    light = None
    motion = None
    exercise = None
    skin_color = None
    gender = None
    glasser = None
    hair_cover = None
    makeup = None
