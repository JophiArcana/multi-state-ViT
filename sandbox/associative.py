#%%
import sys
sys.path.append("/workspace/multi-state-ViT")
import gc
from typing import Any, Dict, List

import datasets
import einops
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=400)

    base_model_name = "facebook/dino-vitb16"

    # SECTION: Configure model
    from transformers import ViTModel, ViTImageProcessor
    model = ViTModel.from_pretrained(base_model_name).to(DEVICE)
    print(model)
      
    # SECTION: Set up dataset
    from torch.utils.data import DataLoader
    ds = datasets.load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
    
    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
        
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images, labels = zip(*map(dict.values, batch))
        return {
            "image": image_processor(images=(*map(process_grayscale, images),), return_tensors="pt")["pixel_values"].to(DEVICE),
            "label": torch.tensor(labels)
        }

    batch_size = 64
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=torch.Generator(device=DEVICE))

    # SECTION: Debug pattern
    from argparse import Namespace
    from model.saccadic_encoder.predictor.modeling_predictor import SaccadicViTMultiStatePattern
    
    pattern_config = Namespace(
        num_patterns={1: 1024, 2: 1024,},
        hidden_size=model.config.hidden_size,
        covariance_dim=64,
        beam_size=16,
        log_covariance_shift=math.log(0.1),
    )
    P = SaccadicViTMultiStatePattern(pattern_config)
    print(P)
    
    test_features = torch.randn((7, 5, 768,))
    print(P.match(
        hidden_states=torch.randn((7, 0, 768,)),
        context_states=test_features, max_wildcards=0,
    ))
    
    
    raise Exception()


    # SECTION: Configure training
    positional_decoder = nn.Linear(model.config.hidden_size, 2)

    model.eval()
    positional_decoder.train()
    
    optimizer = torch.optim.AdamW(positional_decoder.parameters(), lr=1e-2)
    target = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, 14),
        torch.linspace(-1.0, 1.0, 14),
    ), dim=-1).flatten(0, 1)
    
    colors = torch.stack(torch.meshgrid(
        torch.linspace(0.2, 1.0, 14),
        torch.linspace(0.2, 1.0, 14),
    ) + (torch.zeros((14, 14)),), dim=-1).flatten(0, 1)
    
    
    for it, batch in enumerate(dl):
        pixel_values: torch.Tensor = batch["image"]
        
        with torch.no_grad():
            features: torch.Tensor = model(pixel_values).last_hidden_state[..., 1:, :]
        
        output_positions: torch.Tensor = positional_decoder(features)
        loss = torch.mean((output_positions - target) ** 2)
        
        if it % 5 == 0:
            print("=" * 120)
            print(f"Iteration {it} --- loss: {loss.item()}")
            
            def normalize_im(im: torch.Tensor) -> torch.Tensor:
                min_rgb = torch.min(im.flatten(0, -2), dim=0).values
                max_rgb = torch.max(im.flatten(0, -2), dim=0).values
                return (im - min_rgb) / (max_rgb - min_rgb)
            
            num_ims = 3
            plt.rcParams["figure.figsize"] = (4.0 * num_ims, 4.0 * 2,)
            fig, axs = plt.subplots(nrows=2, ncols=num_ims)
            
            for i in range(num_ims):
                im_ax: Axes = axs[0, i]
                im_ax.set_aspect("equal")
                im_ax.imshow(normalize_im(einops.rearrange(pixel_values[i], "c h w -> h w c")).numpy(force=True))
                im_ax.axis("off")
                
                sc_ax: Axes = axs[1, i]
                sc_ax.set_aspect("equal")
                sc_ax.scatter(*output_positions[i].mT.numpy(force=True), color=colors.numpy(force=True), s=8)
                sc_ax.plot(
                    (-1.0, 1.0, 1.0, -1.0, -1.0,),
                    (-1.0, -1.0, 1.0, 1.0, -1.0,),
                    color="black", linewidth=2.0, linestyle="--",
                )
                
            plt.show()
            plt.close()
            plt.rcdefaults()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        if it == 100:
            raise Exception()
    





# %%
