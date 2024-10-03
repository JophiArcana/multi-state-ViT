import requests
from PIL import Image

import datasets
import torch
import torch.nn as nn
import torch_cluster
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT

from transformers import ViTPreTrainedModel
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, ViTMAEModel
from transformers.image_processing_utils import BaseImageProcessor

from infrastructure.dataset import DATASETS
from infrastructure.settings import *


if __name__ == "__main__":
    dataset_name, n_classes = DATASETS["Ego"][0]
    dataset = datasets.load_dataset(dataset_name, ignore_verifications=True)
    print(dataset.shape)
    print(dataset["test"][0].values())




