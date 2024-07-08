from . import Cfg


class MyMethodCfg(Cfg):
    continual_pretrain = None
    pretrain = True
    freeze = False
    diffnorm = True
    peft = "adapter"  # None

    style_n_clusters = 8
    style_aug = True
    noise_n_clusters = 8
    noise_aug = True
    aug_p = 0.5
    
    init_epoch = 20
    contin_epoch = 10
    init_lr = 1e-4
    contin_lr = 1e-4
    log_path = "./logs/cl_mymethod.log"
    ckpt_path = ""
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    weight_decay = 5e-5
    warmup_t = 5

    gpu_id = 1
    num_gpus = 1
    input_resolution = 96
    num_frames = 160
    use_amp = False
    model_name = "uniformer_small"

    batch_size = 8
    name = "My method"


class ViplF14(Cfg):
    record = "mtcnn_vipl_s80_train.csv"  # "mtcnn_vipl_s80_train.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    folds = [1, 2, 3, 4]
    task = None
    source = [1, 2, 3]


class ViplF5(Cfg):
    record = "mtcnn_vipl_s80_train.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    folds = [5]
    task = None
    source = [1, 2, 3]


class Pure(Cfg):
    record = "mtcnn_pure_s80.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    split = "train"


class Ubfc(Cfg):
    record = "mtcnn_ubfc_s80.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    split = "train"


class Buaa(Cfg):
    record = "mtcnn_buaa_s80.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    split = "train"
    lux = 6.3


class Mmpd_new(Cfg):
    record = "mtcnn_mmpd_new_s80.csv"
    transforms = ['flip', 'resample', 'gauss', 'illusion', 'crop']
    input_resolution = 96
    num_frames = 160
    split = "train"
    
    light = None
    motion = None
    exercise = None
    skin_color = None
    gender = None
    glasser = None
    hair_cover = None
    makeup = None

