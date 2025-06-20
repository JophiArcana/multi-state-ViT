# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViT model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
from torch import nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    # BaseModelOutputWithPooling,
    # ImageClassifierOutput,
    # MaskedImageModelingOutput,
)
from transformers import ViTModel
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    ModelOutput,
    # replace_return_docstrings,
    # torch_int,
)

from model.predictive_encoder.configuration_spvit import PredictiveViTConfig
from infrastructure import utils
from infrastructure.settings import RUNTIME_MODE


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


PATCH_CONFIG_DOF: Dict[str, int] = {
    "translation": 2,
    # "euclidean": 3,
    "scaling": 3,
    # "similarity": 4,
    "non-uniform-scaling": 4,
    # "non-shear": 5,
    # "affine": 6,
}


class PredictiveViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.patch_embeddings = PredictiveViTPatchEmbeddings(config)
        self.position_encoder = nn.Linear(PATCH_CONFIG_DOF[config.patch_config], config.hidden_size, bias=config.pe_bias)
        self.position_decoder = nn.Linear(config.hidden_size, PATCH_CONFIG_DOF[config.patch_config], bias=config.pe_bias)

        self.cls_token = nn.Parameter(torch.randn((config.hidden_size,)), requires_grad=config.use_cls_token)
        self.prd_token = nn.Parameter(torch.randn((config.hidden_size,)), requires_grad=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def sample_initial(self, shape: Tuple[int, ...]) -> torch.Tensor:
        match self.config.patch_config:
            case "translation" | "scaling" | "non-uniform-scaling":
                match self.config.patch_config_distribution:
                    case "uniform":
                        sample = 2 * torch.rand(shape + (PATCH_CONFIG_DOF[self.config.patch_config],)) - 1
                    case "gaussian" | "sigmoid" | "cubic":
                        sample = torch.randn(shape + (PATCH_CONFIG_DOF[self.config.patch_config],))
                        if self.config.patch_config_distribution == "sigmoid":
                            sample = torch.sigmoid(sample)
                        elif self.config.patch_config_distribution == "cubic":
                            sample = utils.inverse_cubic(sample)
                    case _:
                        raise ValueError(self.config.patch_config_distribution)

                scale = torch.tensor(self.config.patch_config_scale)
                match scale.ndim:
                    case 0:
                        return scale * sample
                    case 2:
                        scale = scale[:PATCH_CONFIG_DOF[self.config.patch_config]]  # float: [? x 2]
                        return scale[:, 0] * sample + scale[:, 1]
                    case _:
                        raise ValueError(scale.ndim)
            case _:
                raise ValueError(self.config.patch_config)

    def latent_to_position(
        self,
        x: torch.Tensor,
        return_orthogonal: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        W = self.position_decoder.weight
        proj = x @ W.mT
        if self.position_decoder.bias:
            y = proj + self.position_decoder.bias
        else:
            y = proj

        match self.config.patch_config_distribution:
            case "uniform" | "sigmoid":
                y = torch.sigmoid(y)
            case "cubic":
                y = utils.inverse_cubic(y)

        if return_orthogonal:
            orthogonal_x = x - proj @ torch.linalg.pinv(W).mT
            return y, orthogonal_x
        else:
            return y,

    def forward(
        self,
        pixel_values: torch.Tensor,     # float: [B... x C x H x W]
        patch_config: torch.Tensor,     # float: [B... x N x ?]
    ) -> torch.Tensor:                  # float: [B... x D]
        patch_embeddings = self.patch_embeddings(pixel_values, patch_config)
        positional_embeddings = self.position_encoder(patch_config)             # [B... x N x D]

        # add positional encoding to each token
        embeddings = patch_embeddings + positional_embeddings

        # concatenate CLS and PRD token
        cls_token = self.cls_token.expand(pixel_values.shape[:-3] + (1, self.config.hidden_size,))  # [B... x 1 x D]
        prd_token = self.prd_token.expand(pixel_values.shape[:-3] + (1, self.config.hidden_size,))  # [B... x 1 x D]
        embeddings = torch.cat((cls_token, embeddings, prd_token,), dim=-2)                         # [B... x (N + 2) x D]

        # add dropout
        embeddings = self.dropout(embeddings)

        return embeddings


class PredictiveViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: PredictiveViTConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        self.image_size = image_size
        self.patch_config = config.patch_config
        self.default_patch_scale = config.default_patch_scale

        self.grid: torch.Tensor = torch.stack(torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch_size),
            torch.linspace(-1.0, 1.0, patch_size),
        ) + (torch.ones((patch_size, patch_size)),), dim=-1).transpose(dim0=-3, dim1=-2)    # float: [P x P x 3]

        num_channels, patch_size, hidden_size = config.num_channels, config.patch_size, config.hidden_size
        self.num_channels = num_channels
        self.batchnorm = nn.BatchNorm1d(hidden_size, affine=False)
        
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),      # float: [B... x 64 x P x P]
            nn.SiLU(),                                                  # float: [B... x 64 x P x P]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),     # float: [B... x 128 x P/2 x P/2]
            nn.SiLU(),                                                  # float: [B... x 128 x P/2 x P/2]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),    # float: [B... x 256 x P/4 x P/4]
            nn.SiLU(),                                                  # float: [B... x 256 x P/4 x P/4]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),    # float: [B... x 512 x P/8 x P/8]
            nn.SiLU(),                                                  # float: [B... x 512 x P/8 x P/8]
            nn.Conv2d(512, 1024, kernel_size=patch_size // 8),          # float: [B... x 1024 x 1 x 1]
            nn.SiLU(),                                                  # float: [B... x 1024 x 1 x 1]
            nn.Flatten(start_dim=-3, end_dim=-1),                       # float: [B... x 1024]
            nn.Linear(1024, hidden_size, bias=True),                    # float: [B... x D]
        )
        
        self.patch_decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024, bias=True),                            # float: [B... x 1024]
            nn.Unflatten(dim=-1, unflattened_size=(1024, 1, 1)),                # float: [B... x 1024 x 1 x 1]
            nn.SiLU(),                                                          # float: [B... x 1024 x 1 x 1]
            nn.ConvTranspose2d(1024, 512, kernel_size=patch_size // 8),         # float: [B... x 512 x P/8 x P/8]
            nn.SiLU(),                                                          # float: [B... x 512 x P/8 x P/8]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # float: [B... x 256 x P/4 x P/4]
            nn.SiLU(),                                                          # float: [B... x 256 x P/4 x P/4]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # float: [B... x 128 x P/2 x P/2]
            nn.SiLU(),                                                          # float: [B... x 128 x P/2 x P/2]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # float: [B... x 64 x P x P]
            nn.SiLU(),                                                          # float: [B... x 64 x P x P]
            nn.ConvTranspose2d(64, num_channels, kernel_size=5, padding=2),     # float: [B... x C x P x P]
        )

    def patch_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[:-3]                              # B...
        _x = x.reshape((-1,) + x.shape[-3:])            # float: [B x C x H x W]
        _embd = self.patch_encoder(_x)                  # float: [B x D]
        embd = _embd.reshape(bsz + _embd.shape[-1:])    # float: [B... x D]

        return embd
    
    def latent_to_patch(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[:-1]                              # B...
        _x = x.reshape((-1,) + x.shape[-1:])            # float: [B x D]
        _embd = self.patch_decoder(_x)                  # float: [B x C x H x W]
        embd = _embd.reshape(bsz + _embd.shape[-3:])    # float: [B... x C x H x W]
        
        return embd

    def position_to_patch(
        self,
        pixel_values: torch.Tensor,     # float: [B... x C x H x W]
        patch_config: torch.Tensor,     # float: [B... x N... x ?]
    ) -> torch.Tensor:
        bsz = pixel_values.shape[:-3]           # int: B...
        nbsz = patch_config.shape[len(bsz):-1]  # int: N...
        
        sample_grid = self.grid_sample_points(patch_config, bbox_only=False)                        # float: [B... x N... x P x P x 2]
        _pixel_values = pixel_values.reshape((-1,) + pixel_values.shape[-3:])                       # float: [B x C x H x W]
        _sample_grid = sample_grid.reshape((_pixel_values.shape[0], -1,) + sample_grid.shape[-3:])  # float: [B x N x P x P x 2]
        _sample_patches = torch.vmap(torch.nn.functional.grid_sample, in_dims=(None, 1), out_dims=(1,))(
            _pixel_values, _sample_grid,
            mode="bicubic", padding_mode="zeros", align_corners=True,
        )                                                                                           # float: [B x N x C x P x P]
        sample_patches = _sample_patches.reshape(bsz + nbsz + _sample_patches.shape[-3:])
        
        return sample_patches

    def forward(
        self,
        pixel_values: torch.Tensor,     # float: [B... x C x H x W]
        patch_config: torch.Tensor,     # float: [B... x N... x ?]
    ) -> torch.Tensor:                  # float: [B... x D]
        num_channels = pixel_values.shape[-3]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        sample_patches = self.position_to_patch(pixel_values, patch_config)        # float: [B... x N... x C x P x P]
        embeddings = self.patch_to_latent(sample_patches)                                # float: [B... x N... x D]
        
        bsz = pixel_values.shape[:-3]           # int: B...
        nbsz = patch_config.shape[len(bsz):-1]  # int: N...
        _embeddings = embeddings.view((utils.prod(bsz), utils.prod(nbsz), -1,)) # float: [B x N x D]
        _embeddings = self.batchnorm(_embeddings.mT).mT                         # float: [B x N x D]
        embeddings = _embeddings.view(bsz + nbsz + (-1,))        

        return embeddings































class PredictiveViTSelfAttention(nn.Module):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, "... n (h d) -> ... h n d", h=self.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.mT
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply manual attention mask
        if attention_mask is not None:
            attention_scores = torch.where(attention_mask[..., None, :, :], attention_scores, -torch.inf)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs @ value_layer
        context_layer = einops.rearrange(context_layer, "... h n d -> ... n (h d)")

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PredictiveViTSdpaSelfAttention(PredictiveViTSelfAttention):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            logger.warning_once(
                "`ViTSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask[..., None, :],
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )
        context_layer = einops.rearrange(context_layer, "... h n hd -> ... n (h hd)")

        return context_layer, None


class PredictiveViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class PredictiveViTAttention(nn.Module):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.attention = PredictiveViTSelfAttention(config)
        self.output = PredictiveViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, attention_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PredictiveViTSdpaAttention(PredictiveViTAttention):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__(config)
        self.attention = PredictiveViTSdpaSelfAttention(config)


class PredictiveViTIntermediate(nn.Module):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class PredictiveViTOutput(nn.Module):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


SPVIT_ATTENTION_CLASSES = {
    "eager": PredictiveViTAttention,
    "sdpa": PredictiveViTSdpaAttention,
}


class PredictiveViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SPVIT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = PredictiveViTIntermediate(config)
        self.output = PredictiveViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class PredictiveViTEncoder(nn.Module):
    def __init__(self, config: PredictiveViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PredictiveViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )




























@dataclass
class BaseModelOutputWithInputs(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        context_lengths (`torch.LongTensor` of shape `(batch_size)`):
            Number of context tokens used for prediction for each image in the batchs
        input_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the input of the first layer of the model.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        
    """

    input_position: torch.FloatTensor = None
    input_hidden_state: torch.FloatTensor = None
    context_lengths: torch.LongTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class PredictiveViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PredictiveViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PredictiveViTEmbeddings", "ViTLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PredictiveViTEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            module.prd_token.data = nn.init.trunc_normal_(
                module.prd_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.prd_token.dtype)


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class PredictiveViTModel(PredictiveViTPreTrainedModel):
    def __init__(self, config: PredictiveViTConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = PredictiveViTEmbeddings(config)
        self.encoder = PredictiveViTEncoder(config)

        self.batchnorm = nn.BatchNorm1d(config.hidden_size, affine=False)
        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        if self.config.pretrained is not None:
            base_model: ViTModel = ViTModel.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained,
                config=self.config,
                ignore_mismatched_sizes=True,
            )
            self.encoder.load_state_dict(base_model.encoder.state_dict())

            cls_token = base_model.embeddings.cls_token.data[0, 0]
            self.embeddings.cls_token.__init__(cls_token, requires_grad=True)

            self._backward_compatibility_gradient_checkpointing()
        else:
            self.post_init()

    def get_input_embeddings(self) -> PredictiveViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def visualize_sample(
        self,
        pixel_values: torch.Tensor,                             # float: [B... x C x H x W]
        output: BaseModelOutputWithInputs,
        meta: Dict[str, torch.Tensor],
        # context_lengths: torch.Tensor,                          # int: [B...]
        # sample_config: torch.Tensor,                            # float: [B... x N x ?]
        # predicted_sample_config: Optional[torch.Tensor] = None, # float: [B... x (N + 1) x ?]
        context_prediction: bool = False,
        query_prediction: bool = False,
        num_ims: int = 3,
    ) -> None:
        def normalize_im(im: torch.Tensor) -> torch.Tensor:
            min_rgb = torch.min(im.flatten(0, -2), dim=0).values
            max_rgb = torch.max(im.flatten(0, -2), dim=0).values
            return (im - min_rgb) / (max_rgb - min_rgb)

        def grid_center(grid: torch.Tensor) -> torch.Tensor:
            return (grid[0, 0] + grid[-1, -1]) / 2

        def plot_bbox(ax: Axes, grid: torch.Tensor, center: bool = False, **kwargs: Any) -> None:
            if center:
                utils.call_func_with_kwargs(ax.scatter, args=(*grid_center(grid).numpy(force=True),), kwargs=kwargs)
            utils.call_func_with_kwargs(ax.plot, args=(*zip(
                grid[0, 0].numpy(force=True),
                grid[0, -1].numpy(force=True),
                grid[-1, -1].numpy(force=True),
                grid[-1, 0].numpy(force=True),
                grid[0, 0].numpy(force=True),
            ),), kwargs=kwargs)
        
        sample_grid = self.embeddings.patch_embeddings.grid_sample_points(output.input_position, bbox_only=True)
        predicted_sample_grid = self.embeddings.patch_embeddings.grid_sample_points(torch.cat((
            meta["predicted_context_position"],
            meta["predicted_query_position"][..., None, :],
        ), dim=-2), bbox_only=True)

        plt.rcParams["figure.figsize"] = (4.0 * num_ims, 4.0,)
        fig, axs = plt.subplots(nrows=1, ncols=num_ims,)
        for i in range(num_ims):
            ax: Axes = axs[i]
            ax.set_aspect("equal")

            im = normalize_im(einops.rearrange(pixel_values[i], "c h w -> h w c"))
            ax.imshow(im.numpy(force=True), extent=(-1.0, 1.0, 1.0, -1.0))

            bbox_kwargs = {"s": 32, "linewidth": 1.5, "linestyle": "--",}
            for j in range(output.context_lengths[i]):
                plot_bbox(ax, sample_grid[i, j], center=True, color="black", **bbox_kwargs,)

                if context_prediction:
                    plot_bbox(ax, predicted_sample_grid[i, j], center=False, color="purple", **bbox_kwargs,)
                    ax.arrow(
                        *grid_center(sample_grid[i, j]).numpy(force=True),
                        *(grid_center(predicted_sample_grid[i, j] - sample_grid[i, j])).numpy(force=True),
                        color="purple", width=0.005, head_width=0.1, length_includes_head=True,
                    )

            if query_prediction:
                plot_bbox(ax, predicted_sample_grid[i, -1], color="red", **bbox_kwargs,)

            ax.set_title(f"Image {i}")

        fig.suptitle("Original images")
        plt.show()
        plt.close()
        
        if (context_prediction and "true_context_patch" in meta) or (query_prediction and "true_query_patch" in meta):
            def compare_patches(
                ax: Axes,
                true_patch: torch.Tensor,
                predicted_patch: torch.Tensor,
                color: str = None,
            ) -> None:
                ax.set_aspect("equal")
                ax.axis("off")
                ax.imshow(
                    normalize_im(einops.rearrange(torch.cat((
                        true_patch,
                        predicted_patch,
                    ), dim=-1), "c h w -> h w c")).numpy(force=True),
                    extent=((-1.0, 1.0, -0.5, 0.5,)),
                )
                if color is not None:
                    ax.plot((-1.0, 1.0, 1.0, -1.0, -1.0), (-0.5, -0.5, 0.5, 0.5, -0.5), color=color, linewidth=4.0,)

            nrows = torch.max(output.context_lengths[:num_ims]).item() + 1
            plt.rcParams["figure.figsize"] = (4.0 * num_ims, 2.0 * nrows,)
            fig, axs = plt.subplots(nrows=nrows, ncols=num_ims)
            for i in range(num_ims):
                for j in range(output.context_lengths[i]):
                    ax: Axes = axs[j, i] if nrows > 1 else axs[i]
                    compare_patches(ax, meta["true_context_patch"][i, j], meta["predicted_context_patch"][i, j], color="purple")
                for j in range(output.context_lengths[i], nrows - 1):
                    ax: Axes = axs[j, i] if nrows > 1 else axs[i]
                    ax.axis("off")

                ax: Axes = axs[nrows - 1, i] if nrows > 1 else axs[i]
                compare_patches(ax, meta["true_query_patch"][i], meta["predicted_query_patch"][i], color="red")
            
            fig.suptitle("Sampled patches")
            plt.show()
            plt.close()

        plt.rcdefaults()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithInputs,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_inputs: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithInputs]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = next(self.embeddings.parameters()).dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        bsz = pixel_values.shape[:-3]                                                                           # int: B...
        # context_lengths = torch.poisson(torch.ones(bsz)).to(torch.int)                                          # int: [B...]
        context_lengths = torch.empty(bsz).geometric_(1 / self.config.expected_context_length).to(torch.int64)  # int: [B...]
        max_context_length = torch.max(context_lengths).item()                                                  # int: max_N

        sample_config = self.embeddings.sample_initial(bsz + (max_context_length,))     # float: [B... x max_N x ?]
        embedding_output = self.embeddings(pixel_values, sample_config)                 # float: [B... x (max_N + 2) x D]

        k_idx = torch.arange(embedding_output.shape[-2])
        attention_mask = (k_idx <= context_lengths[..., None]) | (k_idx == max_context_length + 1)  # bool: [B... x (max_N + ?)]
        if not self.config.use_cls_token:
            attention_mask[..., 0] = False

        encoder_outputs: BaseModelOutput = self.encoder(
            embedding_output,
            attention_mask=attention_mask[..., None, :],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_inputs = embedding_output if output_inputs else None

        sequence_output = encoder_outputs[0]
        sequence_output = self.batchnorm(sequence_output.mT).mT
        # sequence_output = self.layernorm(sequence_output)
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output,)
            # head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutputWithInputs(
            input_position=sample_config,
            input_hidden_state=encoder_inputs,
            context_lengths=context_lengths,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# class ViTPooler(nn.Module):
#     def __init__(self, config: ViTConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()
#
#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output
#
#
# @add_start_docstrings(
#     """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).
#
#     <Tip>
#
#     Note that we provide a script to pre-train this model on custom data in our [examples
#     directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).
#
#     </Tip>
#     """,
#     VIT_START_DOCSTRING,
# )
# class ViTForMaskedImageModeling(ViTPreTrainedModel):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__(config)
#
#         self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
#
#         self.decoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=config.hidden_size,
#                 out_channels=config.encoder_stride**2 * config.num_channels,
#                 kernel_size=1,
#             ),
#             nn.PixelShuffle(config.encoder_stride),
#         )
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, MaskedImageModelingOutput]:
#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
#
#         Returns:
#
#         Examples:
#         ```python
#         >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests
#
#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)
#
#         >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#         >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")
#
#         >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
#         >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
#         >>> # create random boolean mask of shape (batch_size, num_patches)
#         >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
#
#         >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#         >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
#         >>> list(reconstructed_pixel_values.shape)
#         [1, 3, 224, 224]
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
#             raise ValueError(
#                 "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
#                 "the reconstructed image has the same dimensions as the input. "
#                 f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
#             )
#
#         outputs = self.vit(
#             pixel_values,
#             bool_masked_pos=bool_masked_pos,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         # Reshape to (batch_size, num_channels, height, width)
#         sequence_output = sequence_output[:, 1:]
#         batch_size, sequence_length, num_channels = sequence_output.shape
#         height = width = math.floor(sequence_length**0.5)
#         sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
#
#         # Reconstruct pixel values
#         reconstructed_pixel_values = self.decoder(sequence_output)
#
#         masked_im_loss = None
#         if bool_masked_pos is not None:
#             size = self.config.image_size // self.config.patch_size
#             bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
#             mask = (
#                 bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
#                 .repeat_interleave(self.config.patch_size, 2)
#                 .unsqueeze(1)
#                 .contiguous()
#             )
#             reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
#             masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
#
#         if not return_dict:
#             output = (reconstructed_pixel_values,) + outputs[1:]
#             return ((masked_im_loss,) + output) if masked_im_loss is not None else output
#
#         return MaskedImageModelingOutput(
#             loss=masked_im_loss,
#             reconstruction=reconstructed_pixel_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
# @add_start_docstrings(
#     """
#     ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
#     the [CLS] token) e.g. for ImageNet.
#
#     <Tip>
#
#         Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
#         setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
#         position embeddings to the higher resolution.
#
#     </Tip>
#     """,
#     VIT_START_DOCSTRING,
# )
# class ViTForImageClassification(ViTPreTrainedModel):
#     def __init__(self, config: ViTConfig) -> None:
#         super().__init__(config)
#
#         self.num_labels = config.num_labels
#         self.vit = ViTModel(config, add_pooling_layer=False)
#
#         # Classifier head
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=ImageClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
#     )
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, ImageClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.vit(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         logits = self.classifier(sequence_output[:, 0, :])
#
#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#
#         return ImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
