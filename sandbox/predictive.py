import sys
sys.path.append("/workspace/multi-state-ViT")
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
import torchvision.datasets
from matplotlib import pyplot as plt
from PIL import Image

from infrastructure.dataset import DATASETS
from infrastructure.settings import DEVICE


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # torch.manual_seed(11163065284731897840)
    torch.manual_seed(1212)
    # torch.manual_seed(10936251301023808035)
    # print(f"Seed: {torch.seed()}")
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=400)

    dataset_name, n_classes = DATASETS["Common"][1]
    base_model_name = "facebook/dino-vitb8"

    # SECTION: Dataset setup
    dataset = datasets.load_dataset(dataset_name)
    dataset_size = dataset["train"].num_rows

    images = []
    subsample_size = 5
    while len(images) < subsample_size:
        images.append(dataset["train"][torch.randint(0, dataset_size, ()).item()]["image"])

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
    images = [*map(process_grayscale, images)]

    # SECTION: Configure model
    from transformers import ViTModel, ViTImageProcessor
    from model.predictive_encoder import (
        PredictiveViTConfig,
        PredictiveViTModel,
        PredictiveViTTrainingConfig,
        training_loss,
    )

    base_model = ViTModel.from_pretrained(base_model_name)

    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
    inputs = image_processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)

    model = PredictiveViTModel(PredictiveViTConfig(
        _attn_implementation="eager",
        patch_config_scale=[
            [0.7, 0.0],
            [0.7, 0.0],
            [0.5, -1.0],
        ],
        patch_size=64,
        pretrained=None,    # base_model_name,
    ))
    print(model)
    
    # # SECTION: Configure training
    # training_config = PredictiveViTTrainingConfig(
    #     preservation=1.0,
    #     context_prediction=1.0,
    #     query_prediction=1.0,
    #     positional_recovery=1.0,
    #     positional_regularization=1.0,
    # )

    # with torch.no_grad():
    #     model.eval()
    #     output = model.forward(
    #         pixel_values=inputs,
    #         output_inputs=True,
    #         output_hidden_states=False,
    #         output_attentions=False,
    #         return_dict=True,
    #     )
    #     combined_loss, losses, meta = training_loss(inputs, model, output, training_config)
    #     print(meta["predicted_context_position"])
    #     print(meta["predicted_query_position"])
    #     raise Exception()
        
    # SECTION: Set up dataset
    from torch.utils.data import DataLoader
    ds = datasets.load_dataset("ILSVRC/imagenet-1k", split="train")
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images, labels = zip(*map(dict.values, batch))
        return {
            "image": image_processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE),
            "label": torch.tensor(labels)
        }
    
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn, generator=torch.Generator(device=DEVICE))
    
    # SECTION: Configure training
    training_config = PredictiveViTTrainingConfig(
        preservation=0.8,
        context_prediction=0.5,
        query_prediction=1.0,
        positional_recovery=0.0,
        positional_regularization=0.1,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    




