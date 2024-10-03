import itertools
from typing import *

import numpy as np
import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans
from ncut_pytorch import NCUT
from sklearn.cluster import HDBSCAN

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import PushToHubMixin

from infrastructure.settings import DEVICE


class ClusteringConfig(PretrainedConfig):
    def __init__(self, **kwargs: Any):
        super(PushToHubMixin, self).__init__()
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                print(f"Can't set {key} with value {value} for {self}")
                raise err


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


class HDBBoostedSpectralClustering(ClusteringModule):
    def __init__(self, config: ClusteringConfig):
        super().__init__()
        self.ncut_dim = config.ncut_dim
        self.ncut = NCUT(num_eig=self.ncut_dim + 1, device=DEVICE)

        self.hdb = HDBSCAN(allow_single_cluster=True)


    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
    ) -> torch.LongTensor:                  # [bsz x seq_len]
        bsz, N = parent_indices.shape

        # for i in range(np.prod(batch_shape)):
        #     ncut_x_i, _ = self.ncut.fit_transform(x[i])
        #     ncut_x_i = ncut_x_i[:, 1:]
        #
        #     if i < 3:
        #         from sklearn.manifold import TSNE
        #         x_embedded_2d = TSNE(n_components=2).fit_transform(ncut_x_i.numpy(force=True))
        #         x_embedded_3d = TSNE(n_components=3).fit_transform(ncut_x_i.numpy(force=True))
        #
        #         from matplotlib import pyplot as plt
        #
        #         c = (x_embedded_3d - np.min(x_embedded_3d, axis=0)) / np.ptp(x_embedded_3d, axis=0)
        #         plt.scatter(*x_embedded_2d.T, c=c, s=16)
        #
        #         plt.title(f"Image {i}")
        #         plt.legend()
        #         plt.show()
        #
        #         plt.imshow(c.reshape(28, 28, 3))
        #         plt.show()
        # raise Exception()

        with torch.no_grad():
            flattened_x = x.flatten(0, -2)                                                  # [(bsz * N) x D]
            flattened_ncut_x, _ = self.ncut.fit_transform(flattened_x)                      # [(bsz * N) x (N_D + 1)]

            ncut_x = flattened_ncut_x[:, 1:].reshape(bsz, N, self.ncut_dim)                 # [bsz x N x N_D]
            flattened_ncut_x = flattened_ncut_x[:, 1:]

            result = torch.zeros((bsz, N), dtype=torch.long)
            n_parent_clusters = torch.max(parent_indices, dim=-1).values + 1
            for im_idx, (im_parent_indices, im_ncut_x) in enumerate(zip(parent_indices, ncut_x)):
                print(im_idx, im_ncut_x.requires_grad)
                print("\t", im_parent_indices.shape, im_ncut_x.shape)
                im_n_parent_clusters = n_parent_clusters[im_idx].item()
                for parent_cluster_idx in range(im_n_parent_clusters):
                    parent_cluster_indices = im_parent_indices == parent_cluster_idx
                    parent_cluster_features = im_ncut_x[parent_cluster_indices]

                    child_cluster_labels = torch.LongTensor(self.hdb.fit_predict(parent_cluster_features))
                    n_child_clusters = torch.max(child_cluster_labels).item() + 1

                    child_centroid_initializations = torch.zeros((n_child_clusters, self.ncut_dim))
                    for i in range(n_child_clusters):
                        child_centroid_initializations[i] = torch.mean(parent_cluster_features[child_cluster_labels == i], dim=0)

                    print("\t\tn_child_clusters", n_child_clusters)
                    print("\t\tchild_centroid_initializations", child_centroid_initializations.shape)
                    kmeans = KMeans(n_clusters=n_child_clusters)
                    child_cluster_labels = kmeans.fit_predict(parent_cluster_features, centroids=child_centroid_initializations)
                    result[im_idx, parent_cluster_indices] = child_cluster_labels

        # flattened_fps_x, _, _ = torch.svd_lowrank(flattened_ncut_x, q=self.fps_dim)     # [(B * N) x F_D], [F_D], [N_D x F_D]
        # fps_indices = torch_cluster.fps(
        #     flattened_fps_x, ratio=self.fps_ratio,
        #     batch=torch.arange(np.prod(batch_shape)).repeat_interleave(N)
        # ).reshape(*batch_shape, -1) % N                                                 # [B... x (r * N)]
        # fps_x = flattened_fps_x.reshape(*batch_shape, N, self.fps_dim)                  # [B... x N x F_D]


        # from tsne_torch import TorchTSNE as TSNE
        from sklearn.manifold import TSNE
        x_embedded_2d = TSNE(n_components=2).fit_transform(flattened_ncut_x.detach()).reshape(bsz, N, 2)
        x_embedded_3d = TSNE(n_components=3).fit_transform(flattened_ncut_x.detach()).reshape(bsz, N, 3)

        for i, (im_x_2d, im_x_3d) in enumerate(zip(x_embedded_2d, x_embedded_3d)):
            print(i)
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
                print(c.shape)
                # c = color.lab2rgb(c)
                plt.scatter(*im_x_2d.T, c=c, s=16)

                plt.title(f"Image {i}")
                plt.legend()
                plt.show()

                plt.imshow(c.reshape(28, 28, 3))
                plt.show(bbox_inches="tight")

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
    "spectral": HDBBoostedSpectralClustering
}








