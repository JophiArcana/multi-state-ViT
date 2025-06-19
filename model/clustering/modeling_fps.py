from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn
from matplotlib import pyplot as plt

from infrastructure import utils
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

    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
        **kwargs: Any,
    ) -> torch.LongTensor:
        bsz, N = parent_indices.shape

        flattened_x = x.flatten(0, -2)                                                                  # [(bsz * N) x D]
        flattened_ncut_x, _ = self.ncut.fit_transform(flattened_x)                                      # [(bsz * N) x ncut_D]
        
        flattened_pca_x, _, _ = torch.pca_lowrank(flattened_ncut_x, self.config.fps_dim)                # [(bsz * N) x fps_D]
        _, sample1_indices = utils.fps(flattened_pca_x, self.config.fps_sample1)                        # [fps_s1]
        # _, sample1_indices = torch_fpsample.sample(flattened_pca_x, self.config.fps_sample1)            # [fps_s1]
        sample1_ncut_x = flattened_ncut_x[sample1_indices]                                              # [fps_s1 x ncut_D]

        similarity_x = Fn.normalize(sample1_ncut_x, dim=-1) @ Fn.normalize(flattened_ncut_x, dim=-1).mT # [fps_s1 x (bsz * N)]
        mean, std = similarity_x.mean(dim=-1), similarity_x.std(dim=-1)                                 # [fps_s1], [fps_s1]
        normalized_similarity_x = (similarity_x - mean[:, None]) / std[:, None]                         # [fps_s1 x (bsz * N)]

        sample1_pca_x, _, _ = torch.pca_lowrank(normalized_similarity_x, q=self.config.fps_dim)         # [fps_s1 x fps_D]
        _, supersample2_indices = utils.fps(sample1_pca_x, self.config.fps_supersample2)                # [fps_ss2]
        # _, supersample2_indices = torch_fpsample.sample(sample1_pca_x, self.config.fps_supersample2)    # [fps_ss2]
        supersample2_ncut_x = sample1_ncut_x[supersample2_indices]                                      # [fps_ss2 x ncut_D]

        similarity_x = Fn.normalize(supersample2_ncut_x, dim=-1) @ Fn.normalize(flattened_ncut_x, dim=-1).mT    # [fps_ss2 x (bsz * N)]
        score = torch.sum(similarity_x > self.config.cosine_similarity_threshold, dim=-1)               # [fps_ss2]

        score, sample2_indices = torch.topk(score, k=self.config.fps_sample2)                           # [fps_s2], [fps_s2]
        similarity_x = similarity_x[sample2_indices].reshape(self.config.fps_sample2, bsz, N)           # [fps_s2 x bsz x N]

        child_indices = torch.argmax(similarity_x, dim=0)                                               # [bsz x N]
        cluster_histogram = torch.sum(child_indices[..., None] == torch.arange(self.config.fps_sample2), dim=1)     # [bsz x fps_s2]

        plt.hist(
            child_indices.flatten().numpy(force=True),
            bins=torch.arange(self.config.fps_sample2 + 1).numpy(force=True), rwidth=0.8
        )
        plt.plot(score.numpy(force=True), label="score")
        plt.yscale("log")
        plt.title(f"Cluster histogram all images")
        plt.show()

        h, w = 3, 5
        for im_idx in range(bsz):
            im_clusters = torch.unique(child_indices[im_idx])
            if im_idx < 3:
                print("\t", len(im_clusters), "\t", torch.unique(child_indices[im_idx]))
                indices = torch.topk(torch.sum(similarity_x[:, im_idx] > self.config.cosine_similarity_threshold, dim=-1), k=h * w).indices

                hm = similarity_x[indices, im_idx].reshape(-1, 28, 28)
                mask = (similarity_x[indices, im_idx] > self.config.cosine_similarity_threshold).reshape(-1, 28, 28)
                import einops
                
                cluster_im = plt.get_cmap("inferno")(einops.rearrange(hm, "(h w) s1 s2 -> (h s1) (w s2)", h=h, w=w, s1=28, s2=28).numpy(force=True))
                cluster_im[einops.rearrange(mask, "(h w) s1 s2 -> (h s1) (w s2)", h=h, w=w, s1=28, s2=28).numpy(force=True)] = 1
                plt.imshow(cluster_im)
                plt.show()
                
                # plt.imshow(einops.rearrange(hm, "(h w) s1 s2 -> (h s1) (w s2)", h=h, w=w, s1=28, s2=28).numpy(force=True), cmap="inferno")
                # plt.imshow(
                #     torch.ones((h * 28, w * 28, 3)).numpy(force=True),
                #     alpha=einops.rearrange(mask, "(h w) s1 s2 -> (h s1) (w s2)", h=h, w=w, s1=28, s2=28).numpy(force=True)
                # )
                # plt.show()
                
                # plt.imshow(child_indices[im_idx].reshape(28, 28).numpy(force=True))
                # plt.show()

                # plt.hist(child_indices[im_idx], bins=torch.arange(self.config.fps_sample2 + 1), rwidth=0.8)
                # plt.title(f"Cluster histogram image{im_idx}")
                # plt.show()
                #
                # plt.imshow(child_indices[im_idx].reshape(28, 28), cmap="inferno")
                # plt.show()
        print(similarity_x.shape)
        raise Exception()




