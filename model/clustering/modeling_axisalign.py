from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn
import torch_fpsample
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT

from infrastructure import utils
from infrastructure.settings import DEVICE
from model.clustering.modeling import ClusteringConfig, ClusteringModule


@dataclass
class AxisAlignClusteringConfig(ClusteringConfig):
    model_type: str = "axis"


class AxisAlignClustering(ClusteringModule):
    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
        **kwargs: Any,
    ) -> torch.LongTensor:
        bsz, N = parent_indices.shape

        flattened_x = x.flatten(0, -2)                                                                  # [(bsz * N) x D]
        flattened_demeaned_x = flattened_x - torch.mean(flattened_x, dim=0)                             # [(bsz * N) x D]
        flattened_ncut_x, _ = self.ncut.fit_transform(flattened_demeaned_x)                             # [(bsz * N) x ncut_D]
        ncut_x = flattened_ncut_x.unflatten(0, (bsz, N))                                                # [bsz x N x ncut_D]
        
        indices = torch.nn.functional.gumbel_softmax(ncut_x, tau=kwargs["temperature"], hard=True)      # [bsz x N x ncut_D]
        attention_mask = indices @ indices.mT                                                           # [bsz x N x N]




