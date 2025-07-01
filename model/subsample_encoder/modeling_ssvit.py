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

import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
from tensordict import TensorDict
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as Fn

from transformers import AutoModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    ImageClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from ..base_encoder.modeling_base import BaseViTEncoder
from ..subsample_encoder.configuration_ssvit import SubsampleViTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


class SubsampleViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: SubsampleViTConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = SubsampleViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = config.hidden_size
        self.grid_size = config.grid_size
        self.patch_size = config.patch_size
        self.config = config

        c = 1 / self.patch_size
        self.patch_points: torch.Tensor = torch.stack(torch.meshgrid(
            torch.linspace(-1.0 + c, 1.0 - c, self.patch_size),
            torch.linspace(-1.0 + c, 1.0 - c, self.patch_size),
        ) + (torch.ones((self.patch_size, self.patch_size)),), dim=-1)  # float: [P x P x 3]

        l = torch.linspace(0.0, 1.0, self.grid_size + 1)                # float: [G + 1]
        g = torch.stack((
            torch.stack(torch.meshgrid(l[:-1], l[:-1]), dim=-1),        # float: [G x G x 2(x, y)]
            torch.stack(torch.meshgrid(l[1:], l[1:]), dim=-1),          # float: [G x G x 2(x, y)]
        ), dim=-2).flatten(0, 1)                                        # float: [G^2 x 2(outer_new) x 2(x, y)]
        self.grid_points = torch.stack((g, 1 - g), dim=-2)              # float: [G^2 x 2(outer_new) x 2(outer_old) x 2(x, y)]

    def grid_sample_points(
        self,
        patch_config: torch.Tensor,         # [B... x N... x 2(outer_old) x 2(x, y)] - ((x0, y0), (x1, y1))
    ) -> Tuple[torch.Tensor, torch.Tensor]: # [B... x N... x G^2 x P x P x 2]
        grid_config = torch.sum(patch_config[..., None, None, :, :] * self.grid_points, dim=-2) # float: [B... x N... x G^2 x 2(outer_new) x 2(x, y)]
        t = torch.mean(grid_config, dim=-2, keepdim=True)                           # float: [B... x N... x G^2 x 1 x 2]
        D = torch.diag_embed((grid_config[..., 1, :] - grid_config[..., 0, :]) / 2) # float: [B... x N... x G^2 x 2 x 2]
        affine_transform = torch.cat((D, t), dim=-2)                                # float: [B... x N... x G^2 x 3 x 2]
        return grid_config, self.grid @ affine_transform[..., None, :, :]           # float: [B... x N... x G^2 x P x P x 2]

    def interpolate_pos_encoding(
        self,
        sample_grid: torch.Tensor,  # float: [B... x 2]
    ) -> torch.Tensor:              # float: [B... x D]
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        patch_pos_embed = self.position_embeddings[0, 1:]
        k = int(patch_pos_embed.shape[0] ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape((k, k, self.hidden_size,))    # float: [h x w x D]
        patch_pos_embed = einops.rearrange(patch_pos_embed, "h w c -> c h w")   # float: [D x h x w]

        bsz = sample_grid.shape[:-1]
        _sample_grid = sample_grid.reshape((-1, 2))                             # float: [B x 2]
        _embedding = Fn.grid_sample(
            patch_pos_embed[None], torch.flip(_sample_grid[None, :, None, :], dims=(-1,)),
            mode="bicubic", padding_mode="zeros", align_corners=False,
        )[0, :, 0, :]                                                           # float: [B x D]
        embedding = _embedding.reshape(bsz + (self.hidden_size,))               # float: [B... x D]

        return embedding

    def forward(
        self,
        pixel_values: torch.Tensor,         # float: [B... x C x H x W]
        patch_config: torch.Tensor,         # float: [B... x N... x 2 x 2] - ((x0, y0), (x1, y1))
        return_pixel_values: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: # float: [B... x N... x G^2 x 2 x 2], [B... x N... x G^2 x D], [B... x N... x G^2 x C x P x P]
        # Sample the patches from the image
        corners, sample_grid = self.grid_sample_points(patch_config)                                # float: [B... x N... x G^2 x 2 x 2], [B... x N... x G^2 x P x P x 2]

        bsz = pixel_values.shape[:-3]           # int: (B...,)
        nbsz = sample_grid.shape[len(bsz):-3]   # int: (N..., G^2)

        _pixel_values = pixel_values.reshape((-1,) + pixel_values.shape[-3:])                       # float: [B x C x H x W]
        _sample_grid = sample_grid.reshape((_pixel_values.shape[0], -1,) + sample_grid.shape[-3:])  # float: [B x NG^2 x P x P x 2]
        _sample_patches = torch.vmap(Fn.grid_sample, in_dims=(None, 1), out_dims=(1,))(
            _pixel_values, torch.flip(_sample_grid, dims=(-1,)),
            mode="bicubic", padding_mode="zeros", align_corners=False,
        )                                                                                           # float: [B x NG^2 x C x P x P]
        sample_patches = _sample_patches.reshape(bsz + nbsz + _sample_patches.shape[-3:])           # float: [B... x N... x G^2 x C x P x P]

        # Encode the sampled patches
        _sample_patches = sample_patches.reshape((-1,) + sample_patches.shape[-3:])                 # float: [BNG^2 x C x P x P]
        _embeddings = self.patch_embeddings(_sample_patches)                                        # float: [BNG^2 x D]
        embeddings = _embeddings.reshape(bsz + nbsz + _embeddings.shape[-1:])                       # float: [B... x N... G^2 x D]

        # add positional encoding to each token
        centers = (corners[..., 0, :] + corners[..., 1, :]) / 2                                     # float: [B... x N... x G^2 x 2]
        embeddings = embeddings + self.interpolate_pos_encoding(centers)                            # float: [B... x N... x G^2 x D]

        embeddings = self.dropout(embeddings)

        return corners, embeddings, (sample_patches if return_pixel_values else None),


class SubsampleViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: SubsampleViTConfig) -> None:
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        num_patches = (image_size // patch_size) ** 2

        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class SubsampleViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SubsampleViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SubsampleViTEmbeddings", "BaseViTLayer"]
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
        elif isinstance(module, SubsampleViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

    def post_init(self):
        if getattr(self.config, "pretrained", None) is not None:
            base_model: PreTrainedModel = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained,
                config=self.config,
                ignore_mismatched_sizes=True,
            )
            self.load_state_dict(base_model.state_dict())
            self._backward_compatibility_gradient_checkpointing()
        else:
            super().post_init()


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


@dataclass
class BaseModelOutputWithLog(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
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

    last_hidden_state: torch.FloatTensor = None                             # float: [B... x (max_N + 1) x D]
    probability: torch.FloatTensor = None                                   # float: [B...]

    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None           # float: I x (L x [B... x (max_N + 1) x D])
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None              # float: I x (L x [B... x H x (max_N + 1) x (max_N + 1)])

    valid_masks: Optional[Tuple[Tuple[torch.BoolTensor, ...], ...]] = None  # bool: I x [B... x max_N]
    corners: Optional[Tuple[torch.FloatTensor, ...]] = None                 # float: I x [B... x max_N x 2 x 2]
    depths: Optional[Tuple[torch.LongTensor, ...]] = None                   # int: I x [B... x max_N]
    subsample_masks: Optional[Tuple[torch.BoolTensor, ...]] = None          # bool: I x [B... x max_N]
    pixel_values: Optional[Tuple[torch.FloatTensor, ...]] = None            # float: I x [B... x max_N x C x P x P]


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class SubsampleViTModel(SubsampleViTPreTrainedModel):
    def __init__(self, config: SubsampleViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = SubsampleViTEmbeddings(config)
        self.encoder = BaseViTEncoder(config)
        self.projection = nn.Linear(config.hidden_size, 1, bias=True)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> SubsampleViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithLog,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,    # [B... x C x H x W]
        max_depth: int = None,
        tau: float = 1.0,
        **kwargs: bool,
    ) -> BaseModelOutputWithLog:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        _output_keys = {
            k for k, v in BaseModelOutputWithLog.__dataclass_fields__.items()
            if not str(v.type).startswith("typing.Optional")
        }
        kwargs = {
            k: v if v is not None else getattr(self.config, k, False)
            for k, v in kwargs.items() if k in {f"output_{_k}" for _k in _output_keys}
        }

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        bsz: torch.Size = pixel_values.shape[:-3]
        bsz_index = tuple(t[..., None] for t in torch.meshgrid(*map(torch.arange, bsz)))
        cls_token = self.embeddings.cls_token.expand(bsz + (1, self.config.hidden_size,))           # float: [B... x 1 x D]

        g = self.config.grid_size ** 2
        c = -math.log(g)

        initial_config = torch.tensor((
            (-1.0, -1.0,),
            (1.0, 1.0,),
        )).expand(bsz + (2, 2,))                                                                    # float: [B... x 2 x 2]

        VALID_MASK, CORNERS, INPUT_STATE, PREVIOUS_HIDDEN_STATES, LOCK, DEPTH = "vm", "c", "is", "phs", "l", "d"
        PIXEL_VALUES = "pv"

        _corners, _input_state, _pixel_values = self.embeddings.forward(pixel_values, initial_config, kwargs["output_pixel_values"])    # float: [B... x max_N x 2 x 2], [B... x max_N x D]
        T: TensorDict = TensorDict({
            VALID_MASK: torch.full(bsz + (g,), True),   # bool: [B... x max_N]
            CORNERS: _corners,                          # float: [B... x max_N x 2 x 2]
            INPUT_STATE: _input_state,                  # float: [B... x max_N x D]
            PREVIOUS_HIDDEN_STATES: None,               # float: [B... x max_N x L x D]
            LOCK: torch.full(bsz + (g,), False),        # bool: [B... x max_N]
            DEPTH: torch.full(bsz + (g,), 0),           # int: [B... x max_N]
            **({PIXEL_VALUES: _pixel_values} if kwargs["output_pixel_values"] else {}),
        }, batch_size=bsz + (g,)).auto_device_()        # TensorDict: [B... x max_N x ?...]

        def binary_projection(x: torch.Tensor) -> torch.Tensor:
            weights = self.projection.forward(x).squeeze(-1)
            return Fn.gumbel_softmax(
                torch.stack((weights, torch.zeros_like(weights)), dim=-1),
                tau=tau, hard=True, dim=-1,
            )

        it, convergence_mask = 0, torch.tensor(True)
        cumulative_probs = torch.ones(bsz)

        output_log = {k: () for k in _output_keys}
        subsample_mask: torch.Tensor = None
        cls_hidden_states: torch.Tensor = None      # float: [B... x 1 x L x D]
        while it < (max_depth if max_depth is not None else float("inf")) and torch.any(convergence_mask).item():

            def mask_to_indices(
                mask: torch.Tensor, # bool: [B... x max_N]
            ) -> Tuple[torch.Tensor, torch.Tensor, int]:
                mask = mask * T[VALID_MASK]
                count = torch.sum(mask, dim=-1)     # int: [B...]
                max_count = torch.max(count).item()
                weights = mask.to(torch.float) + torch.linspace(0.1, 0.0, mask.shape[-1])
                return torch.topk(weights, k=max_count, dim=-1).indices, count, max_count

            # DONE: Use subsample mask to update corners, input_state, previous_hidden_states, valid_mask, lock, depth
            if subsample_mask is not None:
                subsample_indices, subsample_count, subsample_max_count = mask_to_indices(subsample_mask)   # int: [B... x max_S], [B...], max_S
                subsample_max_count *= g
                new_corners, new_input_state, new_pixel_values = self.embeddings.forward(
                    pixel_values,
                    T[CORNERS][bsz_index + (subsample_indices,)],                       # float: [B... x max_S x 2 x 2]
                    return_pixel_values=kwargs["output_pixel_values"]
                )                                                                       # float: [B... x max_S x G^2 x 2 x 2], [B... x max_S x G^2 x D], [B... x max_S x G^2 x C x P x P]

                T[VALID_MASK] *= ~subsample_mask
                T = TensorDict.cat((T, TensorDict({
                    VALID_MASK: torch.arange(subsample_max_count) < (g * subsample_count)[..., None],   # bool: [B... x (G^2 max_S))]
                    CORNERS: new_corners.flatten(-4, -3),                                               # float: [B... x (G^2 x max_S) x 2 x 2]
                    INPUT_STATE: new_input_state.flatten(-3, -2),                                       # float: [B... x (G^2 max_S) x D]
                    PREVIOUS_HIDDEN_STATES: torch.zeros(bsz + (
                        subsample_max_count,
                        self.config.num_hidden_layers,
                        self.config.hidden_size,
                    )),                                                                                 # float: [B... x (G^2 max_S) x L x D]
                    LOCK: torch.full(bsz + (subsample_max_count,), False),                              # bool: [B... x (G^2 max_S)]
                    DEPTH: torch.repeat_interleave(T[DEPTH][bsz_index + (subsample_indices,)] + 1, g, dim=-1),  # int: [B... x (G^2 max_S)]
                    **({PIXEL_VALUES: new_pixel_values.flatten(-5, -4)} if kwargs["output_pixel_values"] else {}),
                }, batch_size=bsz + (subsample_max_count,)).auto_device_()), dim=-1)


            if (self.config.nesting_mode in ["open", "lock"]) or (T[PREVIOUS_HIDDEN_STATES] is None):
                attention_mask = torch.where(T[VALID_MASK], 0.0, -torch.inf)                            # float: [B... x max_N]
                if self.config.use_weighted_tokens:
                    attention_mask = attention_mask + c * T[DEPTH]
                attention_mask = torch.cat((torch.zeros(bsz + (1,)), attention_mask), dim=-1)           # float: [B... x (max_N + 1)]

                output: BaseModelOutput = self.encoder.forward(
                    torch.cat((cls_token, T[INPUT_STATE]), dim=-2),                                     # float: [B... x (max_N + 1) x D]
                    context_states=None,
                    attention_mask=attention_mask[..., None, None, :],                                  # float: [B... x 1 x 1 x (max_N + 1)]
                    output_attentions=kwargs["output_attentions"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = torch.stack(output.hidden_states, dim=-2)                               # float: [B... x (max_N + 1) x L x D]
                cls_hidden_states = hidden_states[..., :1, :, :]                                        # float: [B... x 1 x L x D]
                T[PREVIOUS_HIDDEN_STATES] = hidden_states[..., 1:, :, :]                                # float: [B... x max_N x L x D]

                mask = binary_projection(output.last_hidden_state)                                      # float: [B... x max_N x 2]
                subsample_mask = (mask[..., 0] == 1.0) * T[VALID_MASK]                                  # bool: [B... x max_N]
                if self.config.nesting_mode in ["lock", "freeze"]:
                    T[LOCK] += (mask[..., 0] == 0.0)

                probs = torch.max(mask, dim=-1).values                                                  # float: [B... x max_N]
                probs = torch.prod(torch.where(T[VALID_MASK] * ~T[LOCK], probs, 1.0), dim=-1)           # float: [B...]

            elif self.config.nesting_mode == "freeze":
                unlocked_indices, unlocked_count, unlocked_max_count = mask_to_indices(~T[LOCK])        # int: [B... x max_U], [B...], max_U
                locked_indices, locked_count, locked_max_count = mask_to_indices(T[LOCK])               # int: [B... x max_L], [B...], max_L

                unlocked_states = T[INPUT_STATE][bsz_index + (unlocked_indices,)]                       # float: [B... x max_U x D]
                locked_states = T[PREVIOUS_HIDDEN_STATES][bsz_index + (locked_indices,)]                # float: [B... x max_L x L x D]

                unlocked_sequence_mask = torch.arange(unlocked_max_count) < unlocked_count[..., None]   # bool: [B... x max_U]
                locked_sequence_mask = torch.arange(locked_max_count) < locked_count[..., None]         # bool: [B... x max_L]

                attention_mask = torch.cat((unlocked_sequence_mask, locked_sequence_mask,), dim=-1)     # bool: [B... x (max_U + max_L)]
                attention_mask = torch.where(attention_mask, 0.0, -torch.inf)                           # float: [B... x (max_U + max_L)]
                if self.config.use_weighted_tokens:
                    attention_mask = attention_mask + c * T[DEPTH][bsz_index + (torch.cat((unlocked_indices, locked_indices,), dim=-1),)]
                attention_mask = torch.cat((torch.zeros(bsz + (1,)), attention_mask), dim=-1)           # float: [B... x (max_U + max_L + 1)]

                output: BaseModelOutput = self.encoder.forward(
                    hidden_states=torch.cat((cls_token, unlocked_states), dim=-2),  # float: [B... x (max_U + max_L + 1) x D]
                    context_states=einops.rearrange(locked_states, "... l d -> l ... d"),
                    attention_mask=attention_mask[..., None, None, :],              # float: [B... x 1 x 1 x (max_U + max_L + 1)]
                    output_attentions=kwargs["output_attentions"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                mask = binary_projection(output.last_hidden_state)                  # float: [B... x max_U x 2]
                subsample_mask = torch.full_like(T[LOCK], False)                    # bool: [B... x max_N]
                subsample_mask[bsz_index + (unlocked_indices,)] = (mask[..., 0] == 1.0) * unlocked_sequence_mask

                hidden_states = torch.stack(output.hidden_states, dim=-2)                               # float: [B... x (max_U + max_L + 1) x L x D]
                cls_hidden_states = hidden_states[..., :1, :, :]                                        # float: [B... x 1 x L x D]
                T[PREVIOUS_HIDDEN_STATES][~T[LOCK]] = hidden_states[..., 1:, :, :][unlocked_sequence_mask]

                _lock = T[LOCK][bsz_index + (unlocked_indices,)]
                _lock = torch.where(unlocked_sequence_mask, mask[..., 0] == 0.0, _lock)
                T[LOCK][bsz_index + (unlocked_indices,)] = _lock

                probs = torch.max(mask, dim=-1).values                              # float: [B... x max_U]
                probs = torch.prod(torch.where(unlocked_sequence_mask, probs, 1.0), dim=-1) # float: [B...]

            else:
                raise ValueError(self.config.nesting_mode)

            # DONE: Log the necessary outputs
            logs = {
                "hidden_states": torch.unbind(torch.cat((cls_hidden_states, T[PREVIOUS_HIDDEN_STATES]), dim=-3), dim=-2),
                "attentions": None,
                "valid_masks": T[VALID_MASK],
                "corners": T[CORNERS],
                "depths": T[DEPTH],
                "subsample_masks": subsample_mask,
                "pixel_values": T[PIXEL_VALUES] if kwargs["output_pixel_values"] else None,
            }
            for k, v in kwargs.items():
                if v:
                    output_log[k] += (logs[k],)

            # DONE: Update the cumulative probabilities
            cumulative_probs = cumulative_probs * probs

            convergence_mask = T[VALID_MASK] * ~T[LOCK]     # bool: [B... x max_N]
            it += 1

        sequence_output = torch.cat((
            cls_hidden_states.squeeze[..., -1, :],          # float: [B... x 1 x D]
            T[PREVIOUS_HIDDEN_STATES][..., -1, :],          # float: [B... x max_N x D]
        ), dim=-2)                                          # float: [B... x (max_N + 1) x D]
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutputWithLog(
            last_hidden_state=sequence_output,
            probability=cumulative_probs,
            **output_log,
        )


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class SubsampleViTForImageClassification(SubsampleViTPreTrainedModel):
    def __init__(self, config: SubsampleViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = SubsampleViTModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit.forward(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
