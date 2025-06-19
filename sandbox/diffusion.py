import sys
sys.path.append("/workspace/multi-state-ViT")

import torch
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline, DiTPipeline
from transformers import ViTMAEForPreTraining, ViTMAEModel

from infrastructure.settings import DEVICE


if __name__ == "__main__":
    # pipe = StableDiffusion3Pipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-3.5-medium",
    #     torch_dtype=torch.float16,
    # )
    # pipe = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",
    # )
    # pipe = DiTPipeline.from_pretrained(
    #     "facebook/DiT-XL-2-256",
    #     torch_dtype=torch.float16,
    # )
    # pipe = pipe.to(DEVICE)
    # print(pipe.vae.decoder)
    
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    print(model)


