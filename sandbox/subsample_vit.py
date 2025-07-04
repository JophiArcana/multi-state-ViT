#%%
import sys
sys.path.append("/workspace/multi-state-ViT")
import gc
from typing import Any, Dict, List

import datasets
import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from infrastructure import utils
from infrastructure.dataset import DATASETS
from infrastructure.settings import DEVICE
from infrastructure.utils import Timer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # torch.manual_seed(11163065284731897840)
    torch.manual_seed(1212)
    # torch.manual_seed(10936251301023808035)
    # print(f"Seed: {torch.seed()}")
    torch.set_printoptions(precision=12, sci_mode=False, linewidth=400)

    base_model_name = "facebook/dinov2-base-imagenet1k-1-layer"

    # SECTION: Configure model
    from transformers import (
        BitImageProcessor,
        Dinov2ForImageClassification,
    )
    from model.subsample_encoder import (
        ImageClassifierOutputWithLog,
        SubsampleViTConfig,
        SubsampleViTForImageClassification,
        visualize_subsample_vit_output,
    )

    image_size = 224
    image_processor = BitImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
 
    config_dict = Dinov2ForImageClassification.from_pretrained(base_model_name).config.to_dict()
    config_dict.update(dict(
        pretrained=base_model_name,
        pretrained_cls=Dinov2ForImageClassification,
        num_hidden_layers=6,
        initial_grid_size=4,
        multiplicative_grid_size=4,
        use_weighted_tokens=True,
        nesting_mode="open",
    ))
    model = SubsampleViTForImageClassification(SubsampleViTConfig(**config_dict))

    # SECTION: Set up dataset
    from torch.utils.data import DataLoader
    ds = datasets.load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
        
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images, labels = zip(*map(dict.values, batch))
        return {
            "pixel_values": image_processor(images=(*map(process_grayscale, images),), return_tensors="pt")["pixel_values"].to(DEVICE),
            "labels": torch.tensor(labels)
        }

    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=torch.Generator(device=DEVICE))

    # SECTION: Set up training
    loss_bn = nn.BatchNorm1d(1, affine=False)
    model.eval()
    model.dinov2.projection.train()
    
    trainable_parameters = model.dinov2.projection.parameters()
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)
    
    max_depth = 1
    for it, batch in enumerate(dl):
        output: ImageClassifierOutputWithLog = model.forward(
            **batch,
            max_depth=max_depth,
            
            output_hidden_states=False,
            output_attentions=False,
            
            output_valid_masks=True,
            output_corners=True,
            output_depths=False,
            output_subsample_logits=True,
            output_subsample_masks=False,
            output_pixel_values=True,
        )
        
        if it % 50 == 0:
            visualize_subsample_vit_output(output)

        
        # normalized_loss = output.loss
        normalized_loss = loss_bn(output.loss[..., None])[..., 0]
        loss = torch.mean(normalized_loss * output.probability)
        if it % 25 == 0:
            print(f"Iteration {it}: normalized loss {loss.item()}")
            print(f"\t--- true loss: {torch.mean(output.loss * output.probability).item()}")
            print(f"\t--- weight: {model.dinov2.projection.weight.norm().item() ** 2}")
            print(f"\t--- bias: {model.dinov2.projection.bias.squeeze().item()}")
            print()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        # if it == 1000:
        #     raise Exception()
    





# %%
