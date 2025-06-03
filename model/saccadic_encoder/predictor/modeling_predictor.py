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
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal, OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.utils.checkpoint
from tensordict import TensorDict
from tensordict.utils import expand_as_right

from transformers.modeling_outputs import (
    ModelOutput,
)

from ..configuration_scvit import SaccadicViTConfig






@dataclass
class BasePatternOutput(ModelOutput):
    complexity: int                         # int: K
    data: TensorDict[str, torch.Tensor]
    # {
    #     pattern_index             - int: [B... x ?],
    #     node_indices              - int: [B... x ? x K],
    #     joint_log_pdf             - float: [B... x ?],
    #     conditional_mean          - float: [B... x ? x (K - k) x D],
    #     conditional_covariance    - float: [B... x ? x (K - k) x D x d],
    # }


class SaccadicViTMultiStatePattern(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.num_patterns = config.num_patterns
        self.hidden_size = config.hidden_size
        self.covariance_dim = config.covariance_dim
        self.beam_size = config.beam_size

        self.mean = nn.ParameterDict({
            str(k): nn.Parameter(torch.randn((v, k, self.hidden_size)), requires_grad=True)     # float: [P x K x D]
            for k, v in self.num_patterns.items()
        })
        self.std = nn.ParameterDict({
            str(k): nn.Parameter(torch.randn((v, k, self.hidden_size, self.covariance_dim)), requires_grad=True)    # float: [P x K x D x d]
            for k, v in self.num_patterns.items()
        })
        self.log_covariance_shift = nn.Parameter(torch.tensor(config.log_covariance_shift), requires_grad=True)     # float
        self.max_k = max(self.num_patterns.keys())

    def match(
        self,
        hidden_states: torch.Tensor,    # float: [B... x N x D]
        context_states: torch.Tensor,   # float: [B... x C x D]
        max_wildcards: int,
    ) -> OrderedDict[Tuple[int, int], BasePatternOutput]:

        bsz = hidden_states.shape[:-2]
        bsz_index = tuple(t[..., None] for t in torch.meshgrid(*map(torch.arange, bsz)))
        wildcard_index: int = -1

        output: OrderedDict[Tuple[int, int], BasePatternOutput] = collections.OrderedDict()
        beam: OrderedDict[int, BasePatternOutput] = collections.OrderedDict([
            (k, BasePatternOutput(
                complexity=k,
                data=TensorDict({
                    "pattern_index": torch.arange(v).expand(bsz + (-1,)),
                    "node_indices": torch.full(bsz + (v, k), wildcard_index),
                    "joint_log_pdf": torch.zeros(bsz + (v,)),
                    "conditional_mean": self.mean[str(k)].expand(bsz + (v, k, self.hidden_size,)),
                    "conditional_covariance": self.std[str(k)].expand(bsz + (v, k, self.hidden_size, self.covariance_dim,)),
                }, batch_size=bsz + (v,)),
            )) for k, v in self.num_patterns.items()
        ])

        def datan_exp_h(t: torch.Tensor) -> torch.Tensor:
            return torch.rsqrt(1 + torch.exp(t))

        def compute_topk_indices(pdf: torch.Tensor, search_dim: int) -> Iterable[torch.Tensor]:
            topk_threshold = torch.topk(pdf.flatten(-search_dim, -1), k=self.beam_size, dim=-1).values[..., -1]     # float: [B...]
            return map(
                lambda t: t.view(bsz + (self.beam_size,)),
                torch.where(pdf >= topk_threshold[(...,) + (None,) * search_dim])[-search_dim:],
            )

        it: int = 0
        while len(beam) > 0:
            new_beam: OrderedDict[int, BasePatternOutput] = collections.OrderedDict()

            for n_remaining, E in beam.items():
                # SECTION: Compute the pairwise PDF between each context token and each marginal by taking the SVD of each marginal component of the covariance
                # On the first step, search only hidden states to guarantee that each element in the final beam
                # contains a match to at least one node in the hidden states
                if it == 0:
                    search_states = hidden_states
                else:
                    search_states = torch.cat((hidden_states, context_states,), dim=-2)

                _pairwise_demean = search_states[..., None, None, :, :] - E.data["conditional_mean"][..., :, :, None, :]    # float: [B... x ? x (K - k) x N x D]
                scaled_pairwise_demean = _pairwise_demean * torch.exp(-0.5 * self.log_covariance_shift)                     # float: [B... x ? x (K - k) x N x D]

                U, S, V = torch.svd(E.data["conditional_covariance"], some=True)                    # float: [B... x ? x (K - k) x {(D x d), (d), (d x d)}]
                log_normalized_L = 2 * torch.log(S) - self.log_covariance_shift                     # float: [B... x ? x (K - k) x d]

                _normalized_U = U * datan_exp_h(-log_normalized_L)[..., None, :]                    # float: [B... x ? x (K - k) x D x d]
                normalized_pairwise_demean = scaled_pairwise_demean @ _normalized_U                 # float: [B... x ? x (K - k) x N x d]

                # TODO: Compute PDF terms
                constant_term = -0.5 * self.hidden_size * torch.log(torch.tensor(2 * torch.pi))     # float: []
                determinant_term = -0.5 * (
                    self.hidden_size * self.log_covariance_shift
                    + torch.sum(torch.log1p(torch.exp(log_normalized_L)), dim=-1)
                )                                                                                   # float: [B... x ? x (K - k)]
                exponent_term = -0.5 * (
                    torch.norm(scaled_pairwise_demean, dim=-1) ** 2
                    - torch.norm(normalized_pairwise_demean, dim=-1) ** 2
                )                                                                                   # float: [B... x ? x (K - k) x N]

                # SECTION: Construct next beam element
                log_pdf = constant_term + determinant_term[..., None] + exponent_term           # float: [B... x ? x (K - k) x N]
                joint_log_pdf = expand_as_right(E.data["joint_log_pdf"], log_pdf) + log_pdf     # float: [B... x ? x (K - k) x ...]

                beam_index, unmatched_slot_index, node_index = compute_topk_indices(joint_log_pdf, 3)   # int: [B... x beam_size]
                beam_data: TensorDict[str, torch.Tensor] = E.data[bsz_index + (beam_index,)]

                # SUBSECTION: Index the pattern index for the new beam
                # DONE: Index the pattern index for the new beam
                beam_pattern_index = beam_data["pattern_index"]                                 # int: [B... x beam_size]

                # SUBSECTION: Update the node indices for the new beam
                beam_node_indices = beam_data["node_indices"]                                   # int: [B... x beam_size x K]
                beam_unmatched_slot_indices = torch.where(beam_node_indices == wildcard_index)[-1].view(bsz + (self.beam_size, n_remaining,))   # int: [B... x beam_size x (K - k)]
                beam_matched_slot_index = torch.gather(beam_unmatched_slot_indices, -1, unmatched_slot_index[..., None])            # int: [B... x beam_size x 1]

                # DONE: Updated the node indices for the new beam
                beam_node_indices[beam_matched_slot_index == torch.arange(E.complexity)] = node_index.flatten()

                # SUBSECTION: Index the conditional joint log density function for the new beam
                # DONE: Indexed conditional joint log density function for the new beam
                beam_joint_log_pdf = joint_log_pdf[bsz_index + (beam_index, unmatched_slot_index, node_index,)]                     # float: [B... x beam_size]

                # TODO: Construct adapter matrix H
                _beam_selected_V = V[bsz_index + (beam_index, unmatched_slot_index,)]                                               # float: [B... x beam_size x d x d]
                _beam_selected_adapter_scale = datan_exp_h(log_normalized_L[bsz_index + (beam_index, unmatched_slot_index)])        # float: [B... x beam_size x d]
                beam_selected_H = _beam_selected_V * _beam_selected_adapter_scale[..., None, :]                                     # float: [B... x beam_size x d x d]

                selected_mask = Fn.one_hot(unmatched_slot_index, num_classes=n_remaining).to(torch.bool)                            # bool: [B... x beam_size x (K - k)]

                # SUBSECTION: Compute conditional covariance for the new beam
                beam_conditional_covariance = beam_data["conditional_covariance"]                                                   # float: [B... x beam_size x (K - k) x D x d]
                beam_unselected_conditional_covariance = beam_conditional_covariance[~selected_mask].unflatten(0, bsz + (self.beam_size, n_remaining - 1,)) # float: [B... x beam_size x (K - k - 1) x D x d]

                # DONE: Computed conditional covariance for the new beam
                beam_unselected_conditional_covariance = beam_unselected_conditional_covariance @ beam_selected_H[..., None, :, :]  # float: [B... x beam_size x (K - k - 1) x D x d]

                # SUBSECTION: Compute conditional mean for the new beam
                beam_conditional_mean = beam_data["conditional_mean"]                                                               # float: [B... x beam_size x (K - k) x D]
                beam_unselected_conditional_mean = beam_conditional_mean[~selected_mask].unflatten(0, bsz + (self.beam_size, n_remaining - 1,)) # float: [B... x beam_size x (K - k - 1) x D]
                beam_selected_normalized_demean = normalized_pairwise_demean[bsz_index + (beam_index, unmatched_slot_index, node_index,)]   # float: [B... x beam_size x d]
                # DONE: Computed conditional mean for the new beam
                beam_unselected_conditional_mean = beam_unselected_conditional_mean + (beam_unselected_conditional_covariance @ beam_selected_normalized_demean[..., None, :, None])[..., 0]    # float: [B... x beam_size x (K - k - 1) x D]

                new_element = BasePatternOutput(
                    complexity=E.complexity,
                    data=TensorDict({
                        "pattern_index": beam_pattern_index,
                        "node_indices": beam_node_indices,
                        "joint_log_pdf": beam_joint_log_pdf,
                        "conditional_mean": beam_unselected_conditional_mean,
                        "conditional_covariance": beam_unselected_conditional_covariance,
                    }, batch_size=bsz + (self.beam_size,)),
                )

                n_remaining -= 1
                if n_remaining <= max_wildcards:
                    output[(E.complexity, n_remaining)] = new_element
                if n_remaining > 0:
                    new_beam[n_remaining] = new_element

            it += 1
            beam = new_beam

        return output


class AbstractSaccadicViTPredictor(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        self.pattern = SaccadicViTMultiStatePattern(config)

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,    # float: [B... x N x D]
        context_states: torch.Tensor,   # float: [B... x C x D]
        patterns: OrderedDict[Tuple[int, int], BasePatternOutput],
        prediction_method: Literal["max", "mean"],
    ) -> Tuple[torch.Tensor, OrderedDict[Tuple[int, int], torch.Tensor]]:
        """"""














