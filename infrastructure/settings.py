import os
import sys
import torch


DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float32
RUNTIME_MODE: str = "debug"
PROJECT_NAME: str = "multi-state-ViT"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
os.chdir(PROJECT_PATH)
sys.path.append(PROJECT_PATH)




