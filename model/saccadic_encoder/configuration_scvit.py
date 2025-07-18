# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""ViT model configuration"""

from types import MappingProxyType
from typing import Callable, Dict, Literal, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SaccadicViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers in the Transformer encoder.
        num_pattern_layers (`int`, *optional*, defaults to 12):
            Number of pattern refinement layers in the Transformer encoder.
        num_patterns (`Dict[int, int]`, *optional*, defaults to {1: 1024, 2: 1024}):
            Number of learned patterns for each complexity.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        pe_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the positional embeddings.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import ViTConfig, ViTModel

    >>> # Initializing a ViT vit-base-patch16-224 style configuration
    >>> configuration = ViTConfig()

    >>> # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
    >>> model = ViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vit"

    def __init__(
        self,
        hidden_size: int = 768,
        num_patterns: Dict[int, int] = MappingProxyType({1: 1024, 2: 1024}),
        covariance_dim: int = 64,
        log_covariance_shift: float = torch.inf,
        beam_size: int = 64,

        refiner_implementation: str = "transformer",

        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[
            Literal["gelu", "relu", "selu", "gelu_new"],
            Callable[[torch.Tensor], torch.Tensor],
        ] = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        qkv_bias: bool = True,

        initializer_range: float = 0.02,
        image_size: int = 448,
        patch_size: int = 128,
        patch_config: str = "translation",
        num_channels: int = 3,
        pe_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_patterns = num_patterns
        self.covariance_dim = covariance_dim
        self.log_covariance_shift = log_covariance_shift
        self.beam_size = beam_size

        self.refiner_implementation = refiner_implementation

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.qkv_bias = qkv_bias

        self.initializer_range = initializer_range
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_config = patch_config
        self.num_channels = num_channels
        self.pe_bias = pe_bias
