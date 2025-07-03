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
        AutoModel,
        BitImageProcessor,
        Dinov2Config,
        Dinov2Model,
        Dinov2ForImageClassification,
    )
    from model.subsample_encoder import (
        BaseModelOutputWithLog,
        ImageClassifierOutputWithLog,
        SubsampleViTConfig,
        SubsampleViTModel,
        SubsampleViTForImageClassification,
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
    def reverse_normalize_im(im):
        return (im * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]

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
            n_ims = 3
            nrows, ncols = 2, (max_depth + 1) * n_ims
            
            plt.rcParams["figure.figsize"] = (2.0 * ncols, 2.0 * nrows,)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
            for im_idx in range(n_ims):
                for depth in range(max_depth + 1):
                    
                    valid_mask = output.valid_masks[depth][im_idx]
                    corners = output.corners[depth][im_idx][valid_mask].numpy(force=True)
                    
                    # Plot nested image
                    pixel_values = np.clip(reverse_normalize_im(einops.rearrange(
                        output.pixel_values[depth][im_idx][valid_mask],
                        "... c h w -> ... h w c",
                    ).numpy(force=True)), a_min=0.0, a_max=1.0)
                    
                    ax_im: Axes = axs[0, im_idx * (max_depth + 1) + depth]
                    ax_im.axis("off")
                    ax_im.set_aspect("equal")
                    for pv, c in zip(pixel_values, corners):
                        ax_im.imshow(pv, extent=(c[0, 1], c[1, 1], c[1, 0], c[0, 0]), interpolation="none",)
                    for c in corners:
                        ax_im.plot(
                            [c[0, 1], c[0, 1], c[1, 1], c[1, 1], c[0, 1],],
                            [c[0, 0], c[1, 0], c[1, 0], c[0, 0], c[0, 0],],
                            color="gold", linewidth=2.0 * (c[1, 0] - c[0, 0]), linestyle="--",
                        )
                    ax_im.set_title(f"Image {im_idx}: depth {depth}")
                    
                    # Plot nested decision tree
                    sigmoid = torch.sigmoid(output.subsample_logits[depth][im_idx][valid_mask]).numpy(force=True)
                    vmin, vmax = 0.4, 0.6   # min(sigmoid), max(sigmoid)
                    
                    ax_dt: Axes = axs[1, im_idx * (max_depth + 1) + depth]
                    ax_dt.axis("off")
                    ax_dt.set_aspect("equal")
                    for s, c in zip(sigmoid, corners):
                        im = ax_dt.imshow(
                            s[None, None],
                            cmap="seismic", vmin=vmin, vmax=vmax, extent=(c[0, 1], c[1, 1], c[1, 0], c[0, 0]), interpolation="none",
                        )
                        
                        center = (c[0] + c[1]) / 2
                        ax_dt.text(
                            center[1], center[0], f"{s:.2f}",
                            fontsize=12.0 * (c[1, 0] - c[0, 0]) ** 0.5, ha="center", va="center",
                        )
                        
                    for c in corners:
                        ax_dt.plot(
                            [c[0, 1], c[0, 1], c[1, 1], c[1, 1], c[0, 1],],
                            [c[0, 0], c[1, 0], c[1, 0], c[0, 0], c[0, 0],],
                            color="black", linewidth=0.2 * (c[1, 0] - c[0, 0]), alpha=0.5,
                        )
                    fig.colorbar(im, cax=make_axes_locatable(ax_dt).append_axes("right", size="5%", pad=0.05), orientation="vertical")

            plt.show()
            plt.close() 
            plt.rcdefaults()
        
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
