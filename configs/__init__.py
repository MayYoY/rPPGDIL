class Cfg:
    name = "Base Config"
    input_resolution = None
    num_frames = None
    num_gpus = None
    gpu_id = None
    model_name = None
    pretrain = None
    
    def __repr__(self) -> str:
        info = f"\n"
        attrs = [i for i in dir(self) if not callable(getattr(self, i))]
        for attr in attrs:
            if attr[0] != '_':
                info += f"\t {attr} = {getattr(self, attr)}\n"
        return info
