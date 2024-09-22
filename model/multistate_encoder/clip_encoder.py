import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel

from model.multistate_encoder.base import MultiStateVisionEncoder, MultiStateEncoderConfig, ClusterAttentionTRMultiStateEncoderBackbone


class CLIPClusterAttentionTRMultiStateVisionEmbeddings(nn.Module):
    def __init__(self, config: MultiStateEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPClusterAttentionTRMultiStateVisionEncoder(MultiStateVisionEncoder):
    def __init__(self, config: MultiStateEncoderConfig):
        pretrained_model = CLIPVisionModel.from_pretrained(config.model_name).vision_model
        super().__init__(
            config,
            CLIPImageProcessor.from_pretrained(config.model_name),
            pretrained_model.embeddings,
            ClusterAttentionTRMultiStateEncoderBackbone(
                config, pretrained_model.encoder.layers
            )
        )




