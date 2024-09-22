import itertools
from argparse import Namespace
from typing import *

import matplotlib.colors
import numpy as np
import torch
import torch.nn as nn
import torch_cluster
from ncut_pytorch import NCUT

from infrastructure.settings import DEVICE


class ClusteringConfig(Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.__dict__.update(kwargs)


class ClusteringModule(nn.Module):
    """
    Args:
        x (torch.FloatTensor): batched tensor of latents to cluster [B... x D]
    
    Returns:
        t (torch.IntTensor): batched tensor of cluster indices [B...] where -1 marks a latent not belonging to any strongly represented cluster
        v (torch.FloatTensor): tensor of cluster means [K x D]
    """
    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
    ) -> torch.LongTensor:
        raise NotImplementedError()


class NCutFPSClustering(ClusteringModule):
    def __init__(self, config: ClusteringConfig):
        super().__init__()
        self.ncut_dim = config.ncut_dim
        self._ncut = NCUT(num_eig=self.ncut_dim + 1, device=DEVICE)

        self.fps_dim = config.fps_dim
        self.fps_ratio = config.fps_ratio

        self.nms_radius = config.nms_radius

    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
    ) -> torch.LongTensor:
        batch_shape = x.shape[:-2]                                                      # B...
        N = x.shape[-2]

        print("Starting NCut")
        flattened_x = x.flatten(0, -2)                                                  # [(B * N) x D]
        flattened_ncut_x, _ = self._ncut.fit_transform(flattened_x)                     # [(B * N) x (N_D + 1)]
        flattened_ncut_x = flattened_ncut_x[:, 1:]                                      # [(B * N) x N_D]
        print("Finished NCut")

        flattened_fps_x, _, _ = torch.svd_lowrank(flattened_ncut_x, q=self.fps_dim)     # [(B * N) x F_D], [F_D], [N_D x F_D]
        fps_indices = torch_cluster.fps(
            flattened_fps_x, ratio=self.fps_ratio,
            batch=torch.arange(np.prod(batch_shape)).repeat_interleave(N)
        ).reshape(*batch_shape, -1) % N                                                 # [B... x (r * N)]
        fps_x = flattened_fps_x.reshape(*batch_shape, N, self.fps_dim)                  # [B... x N x F_D]


        from sklearn.manifold import TSNE
        x_embedded_2d = TSNE(n_components=2).fit_transform(flattened_ncut_x.detach()).reshape(*batch_shape, N, 2)
        x_embedded_3d = TSNE(n_components=3).fit_transform(flattened_ncut_x.detach()).reshape(*batch_shape, N, 3)

        for i, (indices, im_x_2d, im_x_3d) in enumerate(zip(fps_indices, x_embedded_2d, x_embedded_3d)):
            if i < 3:
                from matplotlib import pyplot as plt
                from skimage import color

                # mask = torch.full((len(im_fps_x),), False)
                # mask[indices] = True
                # x_embedded = TSNE(n_components=2).fit_transform(im_fps_x.detach())
                #
                # plt.scatter(*x_embedded[~mask].T, s=4, label="unsampled_points")
                # plt.scatter(*x_embedded[mask].T, s=16, label="sampled_points")
                c = (im_x_3d - np.min(im_x_3d, axis=0)) / np.ptp(im_x_3d, axis=0)
                # c = color.lab2rgb(c)
                plt.scatter(*im_x_2d.T, c=c, s=16)

                plt.title(f"Image {i}")
                plt.legend()
                plt.show()

                plt.imshow(c[1:].reshape(14, 14, 3))
                plt.show()

                print(c.shape)


        raise Exception("Here")





class LatentGMMClustering(ClusteringModule):
    def __init__(self, config: ClusteringConfig):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim

        self.max_clusters = config.max_clusters

        def construct_mlp(dims: List[int]) -> nn.Sequential:
            return nn.Sequential(
                *itertools.chain(*[[
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                ] for i in range(len(dims) - 2)]),
                nn.Linear(dims[-2], dims[-1])
            )

        self.dims = [self.input_dim, self.input_dim // 2, self.input_dim // 4, self.hidden_dim]

        self.encoder = construct_mlp(self.dims)
        self.decoder = construct_mlp([*reversed(self.dims)])

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.IntTensor, torch.FloatTensor]:
        latents = self.encoder(x)
        raise NotImplementedError()


CLUSTERING_CLASSES: Dict[str, type] = {
    "spectral": NCutFPSClustering
}








