from argparse import Namespace
from PIL import Image
from typing import *

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTPreTrainedModel, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.modeling_outputs import BaseModelOutput

from model.clustering import ClusteringConfig


class MultiStateEncoderConfig(PretrainedConfig):
    def __init__(
            self,
            generation_period: int,
            hidden_size: int,
            clustering_config: ClusteringConfig,
            image_size: int = None,
            model_name: str = None,
    ):
        super().__init__()
        self.generation_period = generation_period
        self.hidden_size = hidden_size
        self.cluster = clustering_config
        self.image_size = image_size
        self.model_name = model_name


class MultiStateEncoderBackbone(nn.Module):
    def __init__(self, config: MultiStateEncoderConfig, layers: nn.ModuleList):
        super().__init__()
        self.config = config

        self._vit_layers = layers
        self._cluster_layers = nn.ModuleList([
            self.config.cluster.method(self.config.cluster)
            for _ in range(len(self._vit_layers))
        ])

    def forward(self, hidden_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError()


class MultiStateVisionEncoder(nn.Module):
    def __init__(
            self,
            config: MultiStateEncoderConfig,
            image_processor: BaseImageProcessor,
            embeddings: nn.Module,
            backbone: MultiStateEncoderBackbone,
    ):
        super().__init__()
        self.config = config
        self.image_processor = image_processor
        self.embeddings = embeddings
        self.backbone = backbone

    def forward(self, images: List[Image.Image]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        inputs = self.image_processor(
            images, **({
                "size": {"shortest_edge": self.config.image_size},
                "crop_size": self.config.image_size
            } if self.config.image_size is not None else {}),
            return_tensors="pt"
        )
        return self.backbone(self.embeddings(**inputs))


class ClusterAttentionTRMultiStateEncoderBackbone(MultiStateEncoderBackbone):
    def __init__(self, config: MultiStateEncoderConfig, layers: nn.ModuleList):
        super().__init__(config, layers)
        self.transmitter_embedding = nn.Parameter(torch.randn(self.config.hidden_size))
        self.receiver_embedding = nn.Parameter(torch.randn(self.config.hidden_size))

    def forward(self, hidden_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        all_hidden_states = (hidden_states,)
        n_input_states = hidden_states.shape[0]

        tr_token_pairs = ()
        for layer_idx, (layer_module, cluster_module) in enumerate(zip(self._vit_layers, self._cluster_layers)):
            if layer_idx != 0 and layer_idx % self.config.generation_period == 0:
                input_hidden_states = hidden_states[:n_input_states]                                                # [Ni]
                generated_hidden_states = hidden_states[n_input_states:]                                            # [Ng]

                cluster_indices, cluster_centers = cluster_module(input_hidden_states)                              # [Ni], [K x D]

                augmented_cluster_centers = torch.cat([
                    cluster_centers,                                                                                # [K x D]
                    torch.zeros((1, self.config.hidden_size)),                                                      # [1 x D]
                ], dim=0)                                                                                           # [(K + 1) x D]
                demeaned_input_hidden_states = input_hidden_states - augmented_cluster_centers[cluster_indices]     # [Ni x D]

                present_indices = torch.any(torch.BoolTensor(torch.arange(cluster_indices.max())[:, None] == cluster_indices.flatten()), dim=-1)
                extracted_hidden_states = cluster_centers[torch.where(present_indices)]                             # [dNg x D]

                hidden_states = torch.cat([
                    demeaned_input_hidden_states,                                                                   # [Ni x D]
                    generated_hidden_states,                                                                        # [Ng x D]
                    extracted_hidden_states                                                                         # [dNg x D]
                ], dim=0)                                                                                           # [(Ni + (Ng + dNg)) x D]

            print(f"Starting layer_idx {layer_idx}")
            print(layer_module)
            raise Exception()
            hidden_states = layer_module(hidden_states, attention_mask=None, causal_attention_mask=None)[0]
            print(f"Finished layer_idx {layer_idx}")
            all_hidden_states += (hidden_states,)

        raise NotImplementedError()
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=all_hidden_states,
        # )




