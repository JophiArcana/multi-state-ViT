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

import collections
import collections.abc
from typing import Dict, List, Literal, Optional, OrderedDict, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.models.vit.modeling_vit import ViTAttention
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ..configuration_scvit import SaccadicViTConfig
from ..modeling_quadratic_attention import ViTQuadraticAttention

from .modeling_predictor import (
    AbstractSaccadicViTPredictor,
    BasePatternOutput,
)






class SaccadicViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states






class SaccadicViTAttention(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.attention = ViTQuadraticAttention(config)
        self.output = SaccadicViTSelfOutput(config)
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
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, hidden_states, head_mask=head_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs





class SaccadicViTIntermediate(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
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







class SaccadicViTOutput(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states








SACCADIC_VIT_ATTENTION_CLASSES = {
    "eager": ViTAttention,
    "quadratic": ViTQuadraticAttention,
}




class SaccadicViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SACCADIC_VIT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = SaccadicViTIntermediate(config)
        self.output = SaccadicViTOutput(config)
        # self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            hidden_states,  # self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            attention_mask=attention_mask,
            head_mask=head_mask,
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












class SaccadicViTEncoder(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SaccadicViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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











class TransformerSaccadicViTPredictor(AbstractSaccadicViTPredictor):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__(config)
        self.covariance_dim = config.covariance_dim

        self.cls_tokens = nn.ParameterDict({
            str(k): nn.Parameter(torch.randn((v, self.hidden_size)), requires_grad=True)    # float: [P x D]
            for k, v in self.num_patterns.items()
        })
        self.encoder = SaccadicViTEncoder(config)

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        _hidden_states: torch.Tensor,    # float: [B... x N x D]
        context_states: torch.Tensor,   # float: [B... x C x D]
        patterns: OrderedDict[Tuple[int, int], BasePatternOutput],
        prediction_method: Literal["max", "mean"],
    ) -> Tuple[torch.Tensor, OrderedDict[Tuple[int, int], torch.Tensor]]:

        bsz = hidden_states.shape[:-2]
        n_hidden_states = hidden_states.shape[-2]                               # int: N

        cumulative_n_hidden_states = _hidden_states.shape[-2] + context_states.shape[-2]    # int: 
        predicted_states_list: List[torch.Tensor] = []

        pattern_tokens_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        slice_indices: List[int] = [0]
        for (complexity, n_wildcards), P in patterns.items():


            match prediction_method:
                case "max":
                    predicted_states = P.data["conditional_mean"]   # float: [B... x ? x (K - k) x D]
                case "mean":
                    mean: torch.Tensor = P.data["conditional_mean"] # float: [B... x ? x (K - k) x D]
                    explicit_noise = (P.data["conditional_covariance"] @ torch.randn(mean.shape[:-1] + (self.covariance_dim, 1,)))[..., 0]
                    implicit_noise = torch.exp(0.5 * self.pattern.log_covariance_shift) * torch.randn_like(mean)
                    predicted_states = mean + explicit_noise + implicit_noise
                case _:
                    raise ValueError()
















            pattern_tokens_list.append(self.cls_tokens[str(complexity)][P.data["pattern_index"]])   # float: [B... x ? x D]
            attention_mask_list.append(torch.any(P.data["node_indices"][..., None] == torch.arange(n_hidden_states), dim=-2))   # bool: [B... x ? x N]




            slice_indices.append(slice_indices[-1] + P.data.shape[-1])

        pattern_tokens = torch.cat(pattern_tokens_list, dim=-2)                 # float: [B... x +? x D]
        n_pattern_tokens = pattern_tokens.shape[-2]

        attention_mask = torch.cat(attention_mask_list, dim=-2)                 # bool: [B... x +? x N]
        attention_mask = torch.cat((
            torch.cat((torch.full((n_hidden_states, n_hidden_states,), False), attention_mask.mT,), dim=-1),
            torch.cat((attention_mask, torch.eye(n_pattern_tokens).to(torch.bool),), dim=-1)
        ), dim=-2)                                                              # bool: [B... x (N + +?) x (N + +?)]
        all_tokens = torch.cat((hidden_states, pattern_tokens,), dim=-2)        # float: [B... x (N + +?) x D]

        encoded_tokens = self.encoder(all_tokens, attention_mask=attention_mask)    # float: [B... x (N + +?) x D]
        encoded_hidden_states = encoded_tokens[..., :n_hidden_states, :]            # float: [B... x N x D]
        encoded_pattern_tokens = encoded_tokens[..., n_hidden_states:, :]           # float: [B... x +? x D]

        encoded_pattern_tokens_dict: OrderedDict[Tuple[int, int], torch.Tensor] = collections.OrderedDict()
        for i, k in enumerate(patterns.keys()):
            encoded_pattern_tokens_dict[k] = encoded_pattern_tokens[..., slice_indices[i]:slice_indices[i + 1], :]

        return encoded_hidden_states, encoded_pattern_tokens_dict,






        # invalid_pattern_index = torch.iinfo(torch.int64).max
        #
        # predicted_states_list: List[torch.Tensor] = []
        # for (complexity, n_wildcards), P in patterns.items():
        #     n_predictions = P.data.shape[-1] * n_wildcards
        #
        #     valid_bsz_mask: torch.Tensor = (P.data["pattern_index"] != invalid_pattern_index)
        #     valid_data = P.data[valid_bsz_mask]
        #
        #
        #     valid_predicted_states =

