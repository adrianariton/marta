from torch import device, backends
from torch.cuda import is_available
import sys


def get_device():
    if sys.platform == "darwin" and backends.mps.is_available():
        return device("mps")  # Apple GPU
    elif is_available():
        return device("cuda")  # NVIDIA GPU
    else:
        return device("cpu")  # fallback to CPU
