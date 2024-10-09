from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn
import torch_cluster
from fast_pytorch_kmeans import KMeans
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT

from infrastructure.settings import DEVICE
from model.clustering.modeling import ClusteringConfig, ClusteringModule


@dataclass
class FPSClusteringConfig(ClusteringConfig):
    model_type: str = "fps"
    fps_dim: int = None
    fps_sample1: int = None
    fps_sample2: int = None
    fps_supersample2: int = None
    cosine_similarity_threshold: float = None


class FPSClustering(ClusteringModule):
    def __init__(self, config: FPSClusteringConfig):
        super().__init__(config)
        self.config = config
        self.ncut = NCUT(num_eig=self.config.ncut_dim, device=DEVICE)

    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
    ) -> torch.LongTensor:
        bsz, N = parent_indices.shape

        flattened_x = x.flatten(0, -2)                                                                  # [(bsz * N) x D]
        flattened_ncut_x, _ = self.ncut.fit_transform(flattened_x)                                      # [(bsz * N) x ncut_D]

        flattened_pca_x, _, _ = torch.pca_lowrank(flattened_ncut_x, self.config.fps_dim)                # [(bsz * N) x fps_D]
        sample1_indices = torch_cluster.fps(
            flattened_pca_x, ratio=self.config.fps_sample1 / len(flattened_pca_x)
        )                                                                                               # [fps_s1]
        sample1_ncut_x = flattened_ncut_x[sample1_indices]                                              # [fps_s1 x ncut_D]

        similarity_x = Fn.cosine_similarity(
            sample1_ncut_x[:, None, None],
            flattened_ncut_x[None, :, None], dim=-1
        ).squeeze(-1)                                                                                   # [fps_s1 x (bsz * N)]
        mean, std = similarity_x.mean(dim=-1), similarity_x.std(dim=-1)                                 # [fps_s1], [fps_s1]
        normalized_similarity_x = (similarity_x - mean[:, None]) / std[:, None]                         # [fps_s1 x (bsz * N)]

        sample1_pca_x, _, _ = torch.pca_lowrank(normalized_similarity_x, q=self.config.fps_dim)         # [fps_s1 x fps_D]
        supersample2_indices = torch_cluster.fps(
            sample1_pca_x, ratio=self.config.fps_supersample2 / self.config.fps_sample1
        )                                                                                               # [fps_ss2]
        supersample2_ncut_x = sample1_ncut_x[supersample2_indices]                                      # [fps_ss2 x ncut_D]

        similarity_x = Fn.cosine_similarity(
            supersample2_ncut_x[:, None, None],
            flattened_ncut_x[None, :, None], dim=-1
        ).squeeze(-1)                                                                                   # [fps_ss2 x (bsz * N)]
        score = torch.sum(similarity_x > self.config.cosine_similarity_threshold, dim=-1)               # [fps_ss2]

        score, sample2_indices = torch.topk(score, k=self.config.fps_sample2)                           # [fps_s2], [fps_s2]
        similarity_x = similarity_x[sample2_indices].reshape(self.config.fps_sample2, bsz, N)           # [fps_s2 x bsz x N]

        child_indices = torch.argmax(similarity_x, dim=0)                                               # [bsz x N]
        cluster_histogram = torch.sum(child_indices[..., None] == torch.arange(self.config.fps_sample2), dim=1)     # [bsz x fps_s2]

        plt.hist(child_indices.flatten(), bins=torch.arange(self.config.fps_sample2 + 1), rwidth=0.8)
        plt.plot(score, label="score")
        plt.yscale("log")
        plt.title(f"Cluster histogram all images")
        plt.show()

        h, w = 3, 5
        for im_idx in range(bsz):
            im_clusters = torch.unique(child_indices[im_idx])
            print("\t", len(im_clusters), torch.unique(child_indices[im_idx]))
            if im_idx < 3:
                indices = torch.topk(torch.sum(similarity_x[:, im_idx] > self.config.cosine_similarity_threshold, dim=-1), k=h * w).indices

                hm = similarity_x[indices, im_idx].reshape(-1, 28, 28)
                import einops
                plt.imshow(einops.rearrange(hm, "(h w) s1 s2 -> (h s1) (w s2)", h=h, w=w, s1=28, s2=28).numpy(force=True), cmap="inferno")
                plt.colorbar()
                plt.show()

                # plt.hist(child_indices[im_idx], bins=torch.arange(self.config.fps_sample2 + 1), rwidth=0.8)
                # plt.title(f"Cluster histogram image{im_idx}")
                # plt.show()
                #
                # plt.imshow(child_indices[im_idx].reshape(28, 28), cmap="inferno")
                # plt.show()
        print(similarity_x.shape)
        raise Exception()




