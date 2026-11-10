import torch

def get_device(device_id):
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

def checkpoint_path():
    return "./checkpoints/"
