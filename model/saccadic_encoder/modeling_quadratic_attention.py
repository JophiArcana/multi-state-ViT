import math
from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn as nn

from configuration_scvit import (
    SaccadicViTConfig,
)


class ViTQuadraticAttention(nn.Module):
    def __init__(self, config: SaccadicViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.distance = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, "... n (h d) -> ... h n d", h=self.num_attention_heads, d=self.attention_head_size)

    def forward(
        self,
        query_states: torch.Tensor,                     # float: [B... x Nq x D]
        key_states: torch.Tensor,                       # float: [B... x Nk x D]
        attention_mask: Optional[torch.Tensor] = None,  # bool: [B... x Nq x Nk]
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        qdist_layer = self.transpose_for_scores(self.distance(query_states))            # float: [B... x H x Nq x D']
        kdist_layer = self.transpose_for_scores(self.distance(key_states))              # float: [B... x H x Nk x D']

        query_layer = self.transpose_for_scores(self.query(query_states))               # float: [B... x H x Nq x D']
        key_layer = self.transpose_for_scores(self.key(key_states))                     # float: [B... x H x Nk x D']
        value_layer = self.transpose_for_scores(self.value(key_states))                 # float: [B... x H x Nk x D']

        # Take the dot product between "query" and "key" to get the raw attention scores.
        quadratic_attention_scores = -0.5 * torch.cdist(qdist_layer, kdist_layer) ** 2  # float: [B... x H x Nq x Nk]
        linear_attention_scores = query_layer @ key_layer.mT                            # float: [B... x H x Nq x Nk]

        attention_scores = (quadratic_attention_scores + linear_attention_scores) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = torch.where(attention_mask[..., None, :, :], attention_scores, -torch.inf)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = attention_probs @ value_layer                                   # float: [B... x H x Nq x D']
        context_layer = einops.rearrange(context_layer, "... h n d -> ... n (h d)")     # float: [B... x Nq x D]

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs




