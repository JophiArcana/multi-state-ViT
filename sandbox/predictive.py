#%%
import sys
sys.path.append("/workspace/multi-state-ViT")
import gc
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
import torchvision.datasets
from matplotlib import pyplot as plt
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

    base_model_name = "facebook/dino-vitb8"

    # SECTION: Configure model
    from transformers import ViTModel, ViTImageProcessor
    from model.predictive_encoder import (
        BaseModelOutputWithInputs,
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
 
    model = PredictiveViTModel(PredictiveViTConfig(
        _attn_implementation="sdpa",
        use_cls_token=False,
        patch_config="translation",
        default_patch_scale=1 / 3,
        patch_config_scale=[
            [0.7, 0.0],
            [0.7, 0.0],
            [0.5, -1.0],
        ],
        patch_size=64,
        pretrained=None,    # base_model_name,
    ))
    print(model)
      
    # SECTION: Set up dataset
    from torch.utils.data import DataLoader
    ds = datasets.load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)

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

    # SECTION: Configure training
    training_config = PredictiveViTTrainingConfig(
        preservation=0.8,
        # context_prediction=0.5,
        # query_prediction=1.0,
        context_patch_prediction=1.0,
        query_patch_prediction=0.0001,
        positional_recovery=0.0,
        positional_regularization=1.0,
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for it, batch in enumerate(dl):
        pixel_values = batch["image"]
        output: BaseModelOutputWithInputs = model(
            pixel_values=pixel_values,
            output_inputs=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        combined_loss, losses, meta = training_loss(pixel_values, model, output, training_config)
        loss = torch.mean(combined_loss)

        if it % 50 == 0:
            print("=" * 120)
            print(f"Iteration {it} --- loss: {loss.item()}")
            for k, v in losses.items():
                print(f"\t{k}: {torch.mean(v).item()}")

            print(torch.mean(torch.cat((
                meta["predicted_context_position"],
                meta["predicted_query_position"][..., None, :],
            ), dim=-2) ** 2).item() ** 0.5)
            
            model.visualize_sample(
                pixel_values=pixel_values,
                output=output,
                meta=meta,
                context_prediction=True,
                query_prediction=True,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        if it == 500:
            raise Exception()
    





# %%
