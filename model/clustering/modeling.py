from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


@dataclass
class ClusteringConfig(PretrainedConfig):
    model_type: str = None
    ncut_dim: int = None


class ClusteringModule(nn.Module):
    def __init__(self, config: ClusteringConfig) -> None:
        super().__init__()
        self.config = config

    """
    Args:
        parent_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sequence of indices indicating the parent cluster of each token.
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states.

    Returns:
        child_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sequence of indices indicating the child cluster of each token.
    """
    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
    ) -> torch.LongTensor:
        raise NotImplementedError()




