from src import utils, buaa


class BUAAConfig:
    input_path = ""
    img_cache = ""
    gt_cache = ""
    record_path = "./mtcnn_buaa_s80.csv"

    METHOD = "MTCNN"
    CHUNK_LENGTH = 160
    CHUNK_STRIDE = 80
    MODIFY = True
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = -1
    W = 128
    H = 128
    LARGE_FACE_BOX = False
    CROP_FACE = True
    LARGE_BOX_COEF = 1.0
    DO_CHUNK = True


ops = buaa.FramePreprocess(BUAAConfig())
ops.read_process()
