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

import copy
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

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ..base_encoder.modeling_base import BaseViTEncoder
from ..subsample_encoder.configuration_ssvit import SubsampleViTConfig
from ..subsample_encoder.modeling_outputs import BaseModelOutputWithLog, ImageClassifierOutputWithLog


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
        self.patch_size = config.patch_size
        self.config = config

        c = 1 / self.patch_size
        self.patch_points: torch.Tensor = torch.stack(torch.meshgrid(
            torch.linspace(-1.0 + c, 1.0 - c, self.patch_size),
            torch.linspace(-1.0 + c, 1.0 - c, self.patch_size),
        ) + (torch.ones((self.patch_size, self.patch_size)),), dim=-1)  # float: [P x P x 3]

    def grid_sample_points(
        self,
        patch_config: torch.Tensor,         # float: [B... x N... x 2(outer_old) x 2(x, y)] - ((x0, y0), (x1, y1))
        grid_size: int,                     # int: G
    ) -> Tuple[torch.Tensor, torch.Tensor]: # float: [B... x N... x G^2 x P x P x 2]

        l = torch.linspace(1.0, 0.0, grid_size + 1)                                 # float: [G + 1]
        g = torch.stack((
            torch.stack(torch.meshgrid(l[:-1], l[:-1]), dim=-1),                    # float: [G x G x 2(x, y)]
            torch.stack(torch.meshgrid(l[1:], l[1:]), dim=-1),                      # float: [G x G x 2(x, y)]
        ), dim=-2).flatten(0, 1)                                                    # float: [G^2 x 2(outer_new) x 2(x, y)]
        grid_points = torch.stack((g, 1 - g), dim=-2)                               # float: [G^2 x 2(outer_new) x 2(outer_old) x 2(x, y)]
        
        grid_config = torch.sum(patch_config[..., None, None, :, :] * grid_points, dim=-2)  # float: [B... x N... x G^2 x 2(outer_new) x 2(x, y)]
        t = torch.mean(grid_config, dim=-2, keepdim=True)                           # float: [B... x N... x G^2 x 1 x 2]
        D = torch.diag_embed((grid_config[..., 1, :] - grid_config[..., 0, :]) / 2) # float: [B... x N... x G^2 x 2 x 2]
        affine_transform = torch.cat((D, t), dim=-2)                                # float: [B... x N... x G^2 x 3 x 2]
        return grid_config, self.patch_points @ affine_transform[..., None, :, :]   # float: [B... x N... x G^2 x P x P x 2]

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
        )[0, :, :, 0]                                                           # float: [B x D]
        embedding = _embedding.reshape(bsz + (self.hidden_size,))               # float: [B... x D]

        return embedding

    def forward(
        self,
        pixel_values: torch.Tensor,         # float: [B... x C x H x W]
        patch_config: torch.Tensor,         # float: [B... x N... x 2 x 2] - ((x0, y0), (x1, y1))
        grid_size: int,                     # int: G
        return_pixel_values: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: # float: [B... x N... x G^2 x 2 x 2], [B... x N... x G^2 x D], [B... x N... x G^2 x C x P x P]
        # Sample the patches from the image
        corners, sample_grid = self.grid_sample_points(patch_config, grid_size)                     # float: [B... x N... x G^2 x 2 x 2], [B... x N... x G^2 x P x P x 2]

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
        num_channels = pixel_values.shape[-3]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(-3)
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
        if (pretrained := getattr(self.config, "pretrained", None)) is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained)
            config: PretrainedConfig = type(config)(**self.config.to_dict())
            
            cls = getattr(self.config, "pretrained_cls", AutoModel)
            base_model: PreTrainedModel = cls.from_pretrained(
                pretrained_model_name_or_path=pretrained,
                config=config,
                ignore_mismatched_sizes=True,
            )            
            self.load_state_dict(base_model.state_dict(), strict=False)
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
            if str(v.type).startswith("typing.Optional")
        }
        kwargs = {
            k: kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))
            for k in _output_keys
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

        G = self.config.initial_grid_size ** 2
        g = self.config.multiplicative_grid_size ** 2
        c = -math.log(g)

        initial_config = torch.tensor((
            (-1.0, -1.0,),
            (1.0, 1.0,),
        )).expand(bsz + (2, 2,))                                                                    # float: [B... x 2 x 2]

        VALID_MASK, CORNERS, INPUT_STATE, PREVIOUS_HIDDEN_STATES, LOCK, DEPTH = "vm", "c", "is", "phs", "l", "d"
        PIXEL_VALUES, SUBSAMPLE_LOGITS, SUBSAMPLE_MASK = "pv", "sl", "sm"

        _corners, _input_state, _pixel_values = self.embeddings.forward(
            pixel_values=pixel_values,
            patch_config=initial_config,
            grid_size=self.config.initial_grid_size,
            return_pixel_values=kwargs["pixel_values"],
        )                                               # float: [B... x max_N x 2 x 2], [B... x max_N x D]
        
        T: TensorDict = TensorDict({
            VALID_MASK: torch.full(bsz + (G,), True),   # bool: [B... x max_N]
            CORNERS: _corners,                          # float: [B... x max_N x 2 x 2]
            INPUT_STATE: _input_state,                  # float: [B... x max_N x D]
            PREVIOUS_HIDDEN_STATES: None,               # float: [B... x max_N x L x D]
            LOCK: torch.full(bsz + (G,), False),        # bool: [B... x max_N]
            DEPTH: torch.full(bsz + (G,), 0),           # int: [B... x max_N]

            **({SUBSAMPLE_LOGITS: torch.zeros(bsz + (G,))} if kwargs["subsample_logits"] else {}),      # float: [B... x max_N]
            **({SUBSAMPLE_MASK: torch.full(bsz + (G,), False)} if kwargs["subsample_masks"] else {}),   # bool: [B... x max_N]
            **({PIXEL_VALUES: _pixel_values} if kwargs["pixel_values"] else {}),                        # float: [B... x max_N x C x P x P]
        }, batch_size=bsz + (G,)).auto_device_()        # TensorDict: [B... x max_N x ?...]

        def binary_projection(x: torch.Tensor, return_logits: bool = False) -> Tuple[torch.BoolTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
            weights = self.projection.forward(x).squeeze(-1)                    # float: [B...]
            logits = torch.stack((torch.zeros_like(weights), weights), dim=-1)  # float: [B... x 2]
            dist = torch.distributions.categorical.Categorical(logits=logits)
            r = dist.sample().to(torch.bool)                                    # bool: [B...]
            log_prob = dist.log_prob(r)                                         # float: [B...]
            return r, log_prob, (weights if return_logits else None),

        it, convergence_mask = 0, torch.full(bsz + (G,), True)
        cumulative_log_prob = torch.zeros(bsz)

        output_log = {k: () for k in _output_keys}

        cls_hidden_states: torch.Tensor = None      # float: [B... x 1 x L x D]
        for it in range(max_depth + 1):

            def mask_to_indices(
                mask: torch.Tensor, # bool: [B... x max_N]
            ) -> Tuple[torch.Tensor, torch.Tensor, int]:
                mask = mask * T[VALID_MASK]
                count = torch.sum(mask, dim=-1)     # int: [B...]
                max_count = torch.max(count).item()
                weights = mask.to(torch.float) + torch.linspace(0.1, 0.0, mask.shape[-1])
                return torch.topk(weights, k=max_count, dim=-1).indices, count, max_count

            if (self.config.nesting_mode in ["open", "lock"]) or (T[PREVIOUS_HIDDEN_STATES] is None):
                attention_mask = torch.where(T[VALID_MASK], 0.0, -torch.inf)                            # float: [B... x max_N]
                if self.config.use_weighted_tokens:
                    attention_mask = attention_mask + c * T[DEPTH]
                attention_mask = torch.cat((torch.zeros(bsz + (1,)), attention_mask), dim=-1)           # float: [B... x (max_N + 1)]

                output: BaseModelOutput = self.encoder.forward(
                    torch.cat((cls_token, T[INPUT_STATE]), dim=-2),                                     # float: [B... x (max_N + 1) x D]
                    context_states=None,
                    attention_mask=attention_mask[..., None, None, :],                                  # float: [B... x 1 x 1 x (max_N + 1)]
                    output_attentions=kwargs["attentions"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = torch.stack(output.hidden_states, dim=-2)                               # float: [B... x (max_N + 1) x L x D]
                cls_hidden_states = hidden_states[..., :1, :, :]                                        # float: [B... x 1 x L x D]
                T[PREVIOUS_HIDDEN_STATES] = hidden_states[..., 1:, :, :]                                # float: [B... x max_N x L x D]

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
                    hidden_states=torch.cat((cls_token, unlocked_states), dim=-2),                      # float: [B... x (max_U + max_L + 1) x D]
                    context_states=einops.rearrange(locked_states, "... l d -> l ... d"),
                    attention_mask=attention_mask[..., None, None, :],                                  # float: [B... x 1 x 1 x (max_U + max_L + 1)]
                    output_attentions=kwargs["attentions"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                hidden_states = torch.stack(output.hidden_states, dim=-2)                               # float: [B... x (max_U + max_L + 1) x L x D]
                cls_hidden_states = hidden_states[..., :1, :, :]                                        # float: [B... x 1 x L x D]
                T[PREVIOUS_HIDDEN_STATES][convergence_mask] = hidden_states[..., 1:, :, :][unlocked_sequence_mask]

            else:
                raise ValueError(self.config.nesting_mode)

            mask, log_prob, logits = binary_projection(T[PREVIOUS_HIDDEN_STATES][..., -1, :], return_logits=kwargs["subsample_logits"])
            subsample_mask = mask * convergence_mask
            if self.config.nesting_mode in ["lock", "freeze"]:
                T[LOCK] += ~mask
                
            if kwargs["subsample_logits"]:
                T[SUBSAMPLE_LOGITS][convergence_mask] = logits[convergence_mask]
            if kwargs["subsample_masks"]:
                T[SUBSAMPLE_MASK][convergence_mask] = subsample_mask[convergence_mask]
            
            # DONE: Log the necessary outputs
            logs = {
                "hidden_states": torch.unbind(torch.cat((cls_hidden_states, T[PREVIOUS_HIDDEN_STATES]), dim=-3), dim=-2),
                "attentions": None,
                "valid_masks": T[VALID_MASK].clone(),
                "corners": T[CORNERS].clone(),
                "depths": T[DEPTH].clone(),
                "subsample_logits": T[SUBSAMPLE_LOGITS].clone() if kwargs["subsample_logits"] else None,
                "subsample_masks": T[SUBSAMPLE_MASK].clone() if kwargs["subsample_masks"] else None,
                "pixel_values": T[PIXEL_VALUES].clone() if kwargs["pixel_values"] else None,
            }
            for k, v in kwargs.items():
                if v:
                    output_log[k] += (logs[k],)
 
            if it < max_depth: 
                # DONE: Update the cumulative probabilities
                cumulative_log_prob = cumulative_log_prob + torch.sum(convergence_mask * log_prob, dim=-1)
                                
                # DONE: Use subsample mask to update corners, input_state, previous_hidden_states, valid_mask, lock, depth
                subsample_indices, subsample_count, subsample_max_count = mask_to_indices(subsample_mask)   # int: [B... x max_S], [B...], max_S
                if subsample_max_count > 0:
                    subsample_max_count *= g
                    new_corners, new_input_state, new_pixel_values = self.embeddings.forward(
                        pixel_values=pixel_values,
                        patch_config=T[CORNERS][bsz_index + (subsample_indices,)],                          # float: [B... x max_S x 2 x 2]
                        grid_size=self.config.multiplicative_grid_size,
                        return_pixel_values=kwargs["pixel_values"],
                    )                                                                                       # float: [B... x max_S x G^2 x 2 x 2], [B... x max_S x G^2 x D], [B... x max_S x G^2 x C x P x P]

                    T[VALID_MASK] *= ~subsample_mask
                    T = TensorDict.cat((T, TensorDict({
                        VALID_MASK: torch.arange(subsample_max_count) < (g * subsample_count)[..., None],   # bool: [B... x (G^2 max_S))]
                        CORNERS: new_corners.flatten(-4, -3),                                               # float: [B... x (G^2 x max_S) x 2 x 2]
                        INPUT_STATE: new_input_state.flatten(-3, -2),                                       # float: [B... x (G^2 max_S) x D]
                        PREVIOUS_HIDDEN_STATES: torch.zeros(bsz + (
                            subsample_max_count,
                            self.config.num_hidden_layers + 1,
                            self.config.hidden_size,
                        )),                                                                                 # float: [B... x (G^2 max_S) x L x D]
                        LOCK: torch.full(bsz + (subsample_max_count,), False),                              # bool: [B... x (G^2 max_S)]
                        DEPTH: torch.repeat_interleave(T[DEPTH][bsz_index + (subsample_indices,)] + 1, g, dim=-1),  # int: [B... x (G^2 max_S)]
                        
                        **({SUBSAMPLE_LOGITS: torch.zeros(bsz + (subsample_max_count,))} if kwargs["subsample_logits"] else {}),
                        **({SUBSAMPLE_MASK: torch.full(bsz + (subsample_max_count,), False)} if kwargs["subsample_masks"] else {}),
                        **({PIXEL_VALUES: new_pixel_values.flatten(-5, -4)} if kwargs["pixel_values"] else {}),
                    }, batch_size=bsz + (subsample_max_count,)).auto_device_()), dim=-1)

            convergence_mask = T[VALID_MASK] * ~T[LOCK]     # bool: [B... x max_N]        
            if not torch.any(convergence_mask).item():
                break

        sequence_output = torch.cat((
            cls_hidden_states[..., -1, :],                  # float: [B... x 1 x D]
            T[PREVIOUS_HIDDEN_STATES][..., -1, :],          # float: [B... x max_N x D]
        ), dim=-2)                                          # float: [B... x (max_N + 1) x D]
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutputWithLog(
            last_hidden_state=sequence_output,
            last_valid_mask=T[VALID_MASK],
            log_prob=cumulative_log_prob,
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
    def __init__(self,  config: SubsampleViTConfig) -> None:
        super().__init__(config)
        self.config = config

        self.num_labels = config.num_labels
        self.dinov2 = SubsampleViTModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithLog,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        max_depth: int = None,
        tau: float = 1.0,
        **kwargs,
    ) -> ImageClassifierOutputWithLog:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if not kwargs.get("output_depths", False):
            kwargs["output_depths"] = self.config.use_weighted_tokens

        outputs: BaseModelOutputWithLog = self.dinov2(pixel_values, max_depth=max_depth, tau=tau, **kwargs,)
        
        sequence_output = outputs.last_hidden_state     # float: [B... x (N + 1) x D]
        valid_mask = outputs.last_valid_mask            # bool: [B... x N]

        cls_token = sequence_output[..., 0, :]          # float: [B... x D]
        patch_tokens = sequence_output[..., 1:, :]      # float: [B... x N x D]
        if self.config.use_weighted_tokens:
            depth = outputs.depths[-1]                  # int: [B... x N]
            weights = valid_mask * torch.pow(self.config.multiplicative_grid_size, -6.0 * depth)
        else:
            weights = valid_mask
        avg_patch_token = torch.sum(weights[..., None] * patch_tokens, dim=-2) / torch.sum(weights, dim=-1)[..., None]  # float: [B... x D]

        linear_input = torch.cat((cls_token, avg_patch_token), dim=-1)
        logits: torch.Tensor = self.classifier(linear_input)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = Fn.mse_loss(logits, labels, reduction="none")
            elif self.config.problem_type == "single_label_classification":
                loss = Fn.cross_entropy(logits, labels, reduction="none")
            elif self.config.problem_type == "multi_label_classification":
                loss = Fn.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        return ImageClassifierOutputWithLog(
            loss=loss,
            classifier_logits=logits,
            **vars(outputs),
        )
