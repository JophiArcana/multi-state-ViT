import os
import torch


DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float32
PROJECT_NAME: str = "MultiState"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
os.chdir(PROJECT_PATH)




