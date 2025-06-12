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

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

from transformers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class PredictiveViTTrainingConfig:
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        preservation (`float`, *optional*, defaults to 0.0):
            Loss that motivates transformer encoder to preserve its inputs.
        prediction (`float`, *optional*, defaults to 0.0):
            Loss that motivates transformer output to match embedding at predicted location.
        positional_recovery (`float`, *optional*, defaults to 0.0):
            Loss that motivates embedding decoding of position to recover the position that was sampled.
        positional_regularization (`float`, *optional*, defaults to 0.0):
            Loss that motivates embedding decoding of position to be near the origin.
    """

    preservation: float = 0.0
    context_prediction: float = 0.0
    query_prediction: float = 0.0
    positional_recovery: float = 0.0
    positional_regularization: float = 0.0
