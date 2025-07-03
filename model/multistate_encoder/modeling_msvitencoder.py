"""PyTorch Multi-state ViT encoder model."""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings,
    ViTIntermediate,
    ViTModel,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTSelfAttention,
    ViTSdpaSelfAttention,
    ViTSelfOutput,
)
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward

from infrastructure import utils
from model.clustering import CLUSTERING_CLASSES
from model.multistate_encoder.configuration_msvit import MultiStateViTConfig

# General docstring
_CONFIG_FOR_DOC = "MultiStateViTConfig"


class _MultiStateViTEncoderEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: MultiStateViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.FloatTensor, height: int, width: int) -> torch.FloatTensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class MultiStateViTEncoderEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, position and patch embeddings.
    """
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        return super().forward(pixel_values, bool_masked_pos, interpolate_pos_encoding)[:, 1:]


class MultiStateViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: MultiStateViTConfig) -> None:
        ViTSelfAttention.__init__(self, config)
        self.attention_mask_inf = config.attention_mask_inf

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # self.compress_tokens_with_cluster_indices(
        #     hidden_states,
        #     torch.randn(hidden_states.shape[:2]) > 0
        # )
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply manual attention mask
        if attention_mask is not None:
            attention_scores = attention_scores - self.attention_mask_inf * ~attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs)
        return outputs

    def compress_tokens_with_cluster_indices(
        self,
        hidden_states: torch.FloatTensor,                                                           # [bsz x N x D]
        cluster_indices: torch.LongTensor,                                                          # [bsz x N]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bsz, N, D = hidden_states.shape                                                             # bsz, N, D
        n_clusters: int = torch.max(cluster_indices).item() + 1                                     # C

        query_layer = self.transpose_for_scores(self.query(hidden_states))                          # [bsz x n_heads x N x head_dim]
        key_layer = self.transpose_for_scores(self.key(hidden_states))                              # [bsz x n_heads x N x head_dim]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)                           # [bsz x n_heads x N x N]

        # SECTION: Compress the transmitter and receiver attention probabilities
        masks = (cluster_indices[..., None] == torch.arange(n_clusters))                            # [bsz x N x C]
        transmitter_attention_probs = torch.sum((
            attention_probs[..., None] *                                                            # [bsz x n_heads x N x N x 1]
            masks[..., None, None, :, :]                                                            # [bsz x 1 x 1 x N x C]
        ), dim=-2)                                                                                  # [bsz x n_heads x N x C]
        receiver_attention_probs = torch.transpose(torch.sum((
            attention_probs[..., None] *                                                            # [bsz x n_heads x N x N x 1]
            masks[..., None, :, None, :]                                                            # [bsz x 1 x N x 1 x C]
        ), dim=-3) / torch.sum(masks[..., None, :, None, :], dim=-3), dim0=-2, dim1=-1)             # [bsz x n_heads x C x N]

        # SECTION: Solve for transmitter tokens using least squares
        transmitter_attention_scores = utils.multiclass_logits(transmitter_attention_probs)         # [bsz x n_heads x N x C]
        transmitter_attention_scores = transmitter_attention_scores * math.sqrt(self.attention_head_size)

        QmK = query_layer @ self.key.weight.unflatten(0, (self.num_attention_heads, -1))            # [bsz x n_heads x N x D]
        Qmk = query_layer @ self.key.bias.unflatten(0, (self.num_attention_heads, -1))[..., None]   # [bsz x n_heads x N x 1]
        S = transmitter_attention_scores - Qmk                                                      # [bsz x n_heads x N x C]

        Xh = torch.zeros((n_clusters, n_clusters, bsz, self.num_attention_heads, N, D))             # [C x C x bsz x n_heads x N x D]
        Xh[torch.arange(n_clusters), torch.arange(n_clusters)] = QmK
        Xh = einops.rearrange(Xh, "c1 c2 bsz h n d -> bsz (h n c1) (c2 d)")                         # [bsz x (n_heads * N * C) x (C * D)]
        Xc = torch.repeat_interleave(torch.eye(self.num_attention_heads * N), n_clusters, dim=0)    # [(n_heads * N * C) x (n_heads * N)]
        Xc = Xc.expand(bsz, -1, -1)                                                                 # [bsz x (n_heads * N * C) x (n_heads * N)]

        print(Xh.shape, Xc.shape)
        X = torch.cat((Xh, Xc), dim=-1)                                                             # [bsz x (n_heads * N * C) x (C * D + n_heads * N)]
        y = einops.rearrange(S, "bsz h n c -> bsz (h n c) 1")                                       # [bsz x (n_heads * N * C) x 1]

        W = (torch.linalg.pinv(X) @ y)                                                              # [bsz x (C * D + n_heads * N) x 1]
        transmitter_tokens = einops.rearrange(W[:, :n_clusters * D], "bsz (c d) 1 -> bsz c d")      # [bsz x C x D]

        print(hidden_states.shape, transmitter_tokens.shape)

        raise Exception()




class MultiStateViTSdpaSelfAttention(MultiStateViTSelfAttention, ViTSdpaSelfAttention):
    def __init__(self, config: MultiStateViTConfig) -> None:
        MultiStateViTSelfAttention.__init__(self, config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=-self.attention_mask_inf * ~attention_mask,
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer, None


class MultiStateViTAttention(nn.Module):
    def __init__(self, config: MultiStateViTConfig) -> None:
        super().__init__()
        self.attention = MultiStateViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: List[int]) -> None:
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
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        self_outputs = self.attention.forward(hidden_states, attention_mask)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MultiStateViTSdpaAttention(MultiStateViTAttention):
    def __init__(self, config: MultiStateViTConfig) -> None:
        super().__init__(config)
        self.attention = MultiStateViTSdpaSelfAttention(config)


MULTISTATE_VIT_ATTENTION_CLASSES: Dict[str, Callable[[MultiStateViTConfig], MultiStateViTAttention]] = {
    "eager": MultiStateViTAttention,
    "sdpa": MultiStateViTSdpaAttention,
}


class MultiStateViTEncoderLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: MultiStateViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MULTISTATE_VIT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            attention_mask,
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


@dataclass
class MultiStateViTEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

        last_cluster_tokens (`torch.FloatTensor` of shape `(batch_size, padded_num_clusters, 2, hidden_size)`):
            Sequence of cluster tokens at the output of the last layer of the model.
        cluster_indices (`tuple(torch.LongTensor)`, *optional*, returned when `output_cluster_indices=True` is passed or when `config.output_cluster_indices=True`):
            Tuple of `torch.LongTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length)`.

            Cluster indices of each input token to the model at the output of each layer plus the optional initial embedding outputs.
        cluster_tokens (`tuple(torch.FloatTensor)`, *optional*, returned when `output_cluster_tokens=True` is passed or when `config.output_cluster_tokens=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, padded_num_clusters, 2, hidden_size)`.

            Cluster tokens with the transmitter and receiver token corresponding to each cluster of hidden states,
            padded to account for each image having a different number of clusters.


        last_receiver_to_transmitter_attentions (`torch.FloatTensor` of shape `(batch_size, num_heads, padded_num_clusters, padded_num_clusters)`):
            Attentions weights from each receiver token to every transmitter token at the output of the last layer of
            the model.
        intracluster_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_intracluster_attentions=True` is passed or when `config.output_intracluster_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights between tokens of the same cluster after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        transmitter_to_cluster_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_transmitter_to_cluster_attentions=True` is passed or when `config.output_transmitter_to_cluster_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, padded_num_clusters,
            sequence_length)`.

            Attentions weights between transmitter tokens, and tokens of their corresponding clusters after the
            attention softmax, used to compute the weighted average in the self-attention heads.
        cluster_to_receiver_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_cluster_to_receiver_attentions=True` is passed or when `config.output_cluster_to_receiver_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            padded_num_clusters)`.

            Attentions weights between tokens of clusters and their corresponding receiver tokens after the
            attention softmax, used to compute the weighted average in the self-attention heads.
        receiver_to_transmitter_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_receiver_to_transmitter_attentions=True` is passed or when `config.output_receiver_to_transmitter_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, padded_num_clusters,
            padded_num_clusters)`.

            Attentions weights from each receiver token to every transmitter token after the attention softmax, used to
            compute the weighted average in the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] = None

    last_cluster_tokens: torch.FloatTensor = None
    cluster_indices: Tuple[torch.LongTensor, ...] = None
    cluster_tokens: Tuple[torch.FloatTensor, ...] = None

    last_receiver_to_transmitter_attentions: torch.FloatTensor = None
    intracluster_attentions: Tuple[torch.FloatTensor, ...] = None
    transmitter_to_cluster_attentions: Tuple[torch.FloatTensor, ...] = None
    cluster_to_receiver_attentions: Tuple[torch.FloatTensor, ...] = None
    receiver_to_transmitter_attentions: Tuple[torch.FloatTensor, ...] = None


class MultiStateViTEncoderBackbone(nn.Module):
    def __init__(self, config: MultiStateViTConfig) -> None:
        super().__init__()
        self.config = config
        self.transmitter_token = nn.Parameter(torch.randn((config.hidden_size,)))
        self.receiver_token = nn.Parameter(torch.randn((config.hidden_size,)))

        self.layer = nn.ModuleList([MultiStateViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.cluster = nn.ModuleList([
            CLUSTERING_CLASSES[config.clustering_config.model_type](config.clustering_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    @staticmethod
    def _construct_attention_mask_indices(cluster_indices: torch.LongTensor) -> Dict[str, Tuple[torch.LongTensor, ...]]:
        n_clusters: torch.LongTensor = cluster_indices.max(dim=1).values + 1
        max_n_clusters: int = torch.max(n_clusters).item()

        attention_mask_indices = {}
        # Allow attention within the same clusters
        b_idx, qtoken_idx, ktoken_idx = torch.where(
            cluster_indices[:, :, None] == cluster_indices[:, None, :]
        )                                                                                                               # [bsz x seq_len x seq_len]
        attention_mask_indices["intracluster_attentions"] = (b_idx, qtoken_idx + 2 * max_n_clusters, ktoken_idx + 2 * max_n_clusters)

        b_idx, tr_idx, token_idx = torch.where(
            torch.arange(max_n_clusters)[None, :, None] == cluster_indices[:, None, :]
        )                                                                                                               # [bsz x n_clusters x seq_len]
        # Allow attention from transmitters to their corresponding clusters
        attention_mask_indices["transmitter_to_cluster_attentions"] = (b_idx, 2 * tr_idx, token_idx + 2 * max_n_clusters)
        # Allow attention from clusters to their corresponding receivers
        attention_mask_indices["cluster_to_receiver_attentions"] = (b_idx, token_idx + 2 * max_n_clusters, 2 * tr_idx + 1)

        # Allow attention from receivers to transmitters
        b_idx, t_idx, r_idx = torch.where(torch.logical_and(
            torch.arange(max_n_clusters)[None, :, None] < n_clusters[:, None, None],
            torch.arange(max_n_clusters)[None, None, :] < n_clusters[:, None, None],
        ))                                                                                                              # [bsz x n_clusters x n_clusters]
        attention_mask_indices["receiver_to_transmitter_attentions"] = (b_idx, 2 * r_idx + 1, 2 * t_idx)
        return attention_mask_indices

    @staticmethod
    def _construct_attention_mask(cluster_indices: torch.LongTensor) -> torch.BoolTensor:
        bsz, seq_len = cluster_indices.shape
        max_n_clusters: int = torch.max(cluster_indices).item() + 1

        # • Construct the attention mask
        total_seq_len = 2 * max_n_clusters + seq_len
        attention_mask = torch.full((bsz, total_seq_len, total_seq_len), False)                                         # [bsz x (2n_clusters + seq_len) x (2n_clusters + seq_len)]

        attention_mask_indices = MultiStateViTEncoderBackbone._construct_attention_mask_indices(cluster_indices)
        for v in attention_mask_indices.values():
            attention_mask[v] = True
        attention_mask = attention_mask[:, None, :, :]                                                                  # [bsz x 1 x (2n_clusters + seq_len) x (2n_clusters + seq_len)]
        return attention_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **output_kwargs: bool,
    ) -> MultiStateViTEncoderModelOutput:
        bsz, seq_len, embed_dim = hidden_states.shape

        # stack [TX] and [RX] tokens
        cluster_tokens = torch.stack((self.transmitter_token, self.receiver_token), dim=0).expand(bsz, 1, -1, -1)       # [bsz x n_clusters x 2 x embed_dim]
        cluster_indices = torch.zeros((bsz, seq_len)).to(torch.long)                                                    # [bsz x seq_len]
        attention_mask = self._construct_attention_mask(cluster_indices)                                                # [bsz x 1 x (2n_clusters + seq_len) x (2n_clusters + seq_len)]

        output = {
            "hidden_states": (hidden_states,),
            "cluster_indices": (cluster_indices,),
            "cluster_tokens": (cluster_tokens,),
        }

        for i, (layer_module, cluster_module) in enumerate(zip(self.layer, self.cluster)):
            if i >= self.config.pregeneration_period and i % self.config.generation_period == 0:
                # • Generate subcluster indices
                child_cluster_indices: torch.LongTensor = cluster_module(cluster_indices, hidden_states)                # [bsz x seq_len]
                n_child_clusters: torch.LongTensor = child_cluster_indices.max(dim=1).values + 1                        # [bsz x n_clusters]

                # • Compare parent cluster indices with new cluster indices and duplicate parent tokens
                cumulative_n_child_clusters = torch.cumsum(n_child_clusters, dim=1)                                     # [bsz x n_clusters]
                max_n_child_clusters = torch.max(cumulative_n_child_clusters[:, -1]).item()                             # new_n_clusters
                parent_indices = torch.searchsorted(
                    cumulative_n_child_clusters,                                                                        # [bsz x n_clusters]
                    torch.arange(max_n_child_clusters)[None], side="right"                                              # [1 x new_n_clusters]
                )                                                                                                       # [bsz x new_n_clusters]

                idx = (torch.arange(bsz)[:, None], parent_indices)
                new_cluster_tokens = cluster_tokens[idx]                                                                # [bsz x new_n_clusters x 2 x embed_dim]

                # • Construct the attention mask
                attention_mask = MultiStateViTEncoderBackbone._construct_attention_mask(child_cluster_indices)           # [bsz x 1 x (2new_n_clusters + seq_len) x (2new_n_clusters + seq_len)]

                # • Update cluster indices to new cluster indices as well as the hidden states
                cluster_indices = child_cluster_indices                                                                 # [bsz x seq_len]
                cluster_tokens = new_cluster_tokens                                                                     # [bsz x n_clusters x 2 x embed_dim]

            # • Concatenate TX/RX tokens with latents
            concatenated_states = torch.cat((cluster_tokens.flatten(1, 2), hidden_states), dim=1)                       # [bsz x (2n_clusters + seq_len) x embed_dim]
            
            # • Pass concatenated latents into transformer layer and extract hidden states corresponding to image patches
            if self.gradient_checkpointing and self.training:
                attention_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    concatenated_states,
                    attention_mask,
                )
            else:
                attention_outputs = layer_module(concatenated_states, attention_mask)

            concatenated_states = attention_outputs[0]                                                                  # [bsz x (2n_clusters + seq_len) x embed_dim]
            cluster_tokens = concatenated_states[:, :-seq_len].unflatten(1, (-1, 2))                                    # [bsz x n_clusters x 2 x embed_dim]
            hidden_states = concatenated_states[:, -seq_len:]                                                           # [bsz x seq_len x embed_dim]

            layer_output = {
                "hidden_states": hidden_states,
                "cluster_indices": cluster_indices,
                "cluster_tokens": cluster_tokens,
            }

            concatenated_attention = attention_outputs[1]                                                               # [bsz x num_heads x (2n_clusters + seq_len) x (2n_clusters + seq_len)]
            for k, (b_idx, k_idx, q_idx) in MultiStateViTEncoderBackbone._construct_attention_mask_indices(cluster_indices).items():
                layer_output[k] = concatenated_attention[
                    :, :, torch.unique(k_idx)[:, None], torch.unique(q_idx)[None, :]
                ]                                                                                                       # [bsz x num_heads x ? x ?]

            # Concatenate computed outputs to running tuple of per-layer outputs
            for k, v in layer_output.items():
                output[k] = output.get(k, ()) + (v,)

        return MultiStateViTEncoderModelOutput(
            last_hidden_state=hidden_states,
            last_cluster_tokens=cluster_tokens,
            last_receiver_to_transmitter_attentions=output["receiver_to_transmitter_attentions"][-1],
            **{
                k: v for k, v in output.items()
                if output_kwargs.get(f"output_{k}", False)
            }
        )


class MultiStateViTEncoderPooler(nn.Module):
    @staticmethod
    def forward(
        cluster_tokens: torch.FloatTensor,                      # [bsz x padded_num_clusters x 2 x embed_dim]
        receiver_to_transmitter_attentions: torch.FloatTensor,  # [bsz x num_heads x padded_num_clusters x padded_num_clusters]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return cluster_tokens[:, :, 0, :], receiver_to_transmitter_attentions   # [bsz x padded_num_clusters x embed_dim], [bsz x num_heads x padded_num_clusters x padded_num_clusters]


class MultiStateViTEncoderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MultiStateViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MultiStateViTEncoderEmbeddings", "MultiStateViTEncoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        def _init_trunc_normal(t: torch.Tensor) -> None:
            t.data = nn.init.trunc_normal_(
                t.data.to(torch.float32),
                mean=0.0, std=self.config.initializer_range,
            ).to(t.dtype)

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
        elif isinstance(module, MultiStateViTEncoderEmbeddings):
            _init_trunc_normal(module.position_embeddings)
        elif isinstance(module, MultiStateViTEncoderBackbone):
            _init_trunc_normal(module.transmitter_token)
            _init_trunc_normal(module.receiver_token)


MULTISTATE_VIT_ENCODER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MultiStateViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MULTISTATE_VIT_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            
        output_cluster_indices (`bool`, *optional*):
            Whether or not to return the cluster_indices of all attention layers. See `cluster_indices` under returned
            tensors for more detail.
        output_cluster_tokens (`bool`, *optional*):
            Whether or not to return the cluster_tokens of all attention layers. See `cluster_tokens` under returned
            tensors for more detail.
        
        output_intracluster_attentions (`bool`, *optional*):
            Whether or not to return the intracluster attentions tensors of all attention layers. See
            `intracluster_attentions` under returned tensors for more detail.
        output_transmitter_to_cluster_attentions (`bool`, *optional*):
            Whether or not to return the transmitter-to-cluster attentions tensors of all attention layers. See
            `transmitter_to_cluster_attentions` under returned tensors for more detail.
        output_cluster_to_receiver_attentions (`bool`, *optional*):
            Whether or not to return the cluster-to-receiver attentions tensors of all attention layers. See
            `cluster_to_receiver_attentions` under returned tensors for more detail.
        output_receiver_to_transmitter_attentions (`bool`, *optional*):
            Whether or not to return the receiver-to-transmitter attentions tensors of all attention layers. See
            `receiver_to_transmitter_attentions` under returned tensors for more detail.
"""


@dataclass
class MultiStateViTEncoderModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        cluster_tokens (`torch.FloatTensor` of shape `(batch_size, padded_num_clusters, 2, hidden_size)`):
            Sequence of cluster tokens at the output of the last layer of the model.
        receiver_to_transmitter_attentions (`torch.FloatTensor` of shape `(batch_size, num_heads, padded_num_clusters, padded_num_clusters)`):
            Attentions weights from each receiver token to every transmitter token at the output of the last layer of
            the model.
    """

    cluster_tokens: torch.FloatTensor = None
    receiver_to_transmitter_attentions: torch.FloatTensor = None


@add_start_docstrings(
    "The bare multi-state ViT Model transformer outputting raw hidden-states without any specific head on top.",
    MULTISTATE_VIT_ENCODER_START_DOCSTRING,
)
class MultiStateViTEncoderModel(MultiStateViTEncoderPreTrainedModel):
    def __init__(self, config: MultiStateViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = MultiStateViTEncoderEmbeddings(self.config, use_mask_token=use_mask_token)
        self.backbone = MultiStateViTEncoderBackbone(self.config)

        self.pooler = MultiStateViTEncoderPooler() if add_pooling_layer else None

        # Initialize weights and apply final processing
        if self.config.pretrained is not None:
            base_model = ViTModel.from_pretrained(self.config.pretrained)
            self.embeddings.load_state_dict(base_model.embeddings.state_dict())
            self.backbone.layer.load_state_dict(base_model.encoder.blocks.state_dict())
            
            cls_token = base_model.embeddings.cls_token.data[0, 0]
            self.backbone.transmitter_token.__init__(cls_token)
            self.backbone.receiver_token.__init__(cls_token)
            
            self._backward_compatibility_gradient_checkpointing()
        else:
            self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.backbone.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MULTISTATE_VIT_ENCODER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        output_type=MultiStateViTEncoderModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ) -> Union[MultiStateViTEncoderModelOutput, MultiStateViTEncoderModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)
            
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        backbone_outputs = self.backbone.forward(embedding_output)

        if self.pooler is None:
            return backbone_outputs
        else:
            last_cluster_tokens, last_receiver_to_transmitter_attentions = self.pooler(
                backbone_outputs.last_cluster_tokens,
                backbone_outputs.last_receiver_to_transmitter_attentions
            )
            return MultiStateViTEncoderModelOutputWithPooling(
                cluster_tokens=last_cluster_tokens,
                receiver_to_transmitter_attentions=last_receiver_to_transmitter_attentions
            )

