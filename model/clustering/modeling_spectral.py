from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as Fn
from cuml.cluster import HDBSCAN, KMeans
# from fast_pytorch_kmeans import KMeans
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT, kway_ncut
from sklearn.manifold import TSNE

from infrastructure.settings import DEVICE
from model.clustering.modeling import ClusteringConfig, ClusteringModule



class HDBNCUT(NCUT):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.cluster_size_threshold = 0.01
        
    def fit_transform(self, features: torch.FloatTensor, precomputed_sampled_indices: torch.LongTensor = None):
        torch.seed()
        U, S, V = torch.pca_lowrank(features, q=8)
        pca_x = U * S
        
        labels = torch.tensor(HDBSCAN(
            min_cluster_size=int(self.cluster_size_threshold * features.shape[0]),
            min_samples=512
        ).fit_predict(pca_x), dtype=torch.long)
        
        counts = [torch.sum(labels == i).item() for i in range(torch.max(labels) + 1)]
        print(torch.unique(labels), f"{counts}/{torch.numel(labels)}")
        clustered_indices = torch.where(labels != -1)[0]
        sampled_indices = clustered_indices[torch.randperm(len(clustered_indices))[:self.num_sample]]
        print(sampled_indices.shape)
        return super().fit_transform(features, precomputed_sampled_indices)


@dataclass
class SpectralClusteringConfig(ClusteringConfig):
    model_type: str = "spectral"
    ncut_dist: Literal["rbf", "cosine"] = None
    eigenvalue_threshold: float = None
    cluster_size_threshold: float = None


class SpectralClustering(ClusteringModule):
    def __init__(self, config: SpectralClusteringConfig):
        super().__init__()
        self.config = config
        self.ncut = NCUT(
            num_eig=self.config.ncut_dim,
            sample_method="random",
            num_sample=10000,
            distance=self.config.ncut_dist,
            affinity_focal_gamma=3.0,
            device=DEVICE
        )
        self.iterative_ncut = NCUT(
            num_eig=self.config.ncut_dim,
            # sample_method="random",
            num_sample=10000,
            distance="cosine",
            affinity_focal_gamma=3.0,
            device=DEVICE
        )


    def forward(
        self,
        parent_indices: torch.LongTensor,   # [bsz x seq_len]
        x: torch.FloatTensor,               # [bsz x seq_len x embed_dim]
        **kwargs: Any,
    ) -> torch.LongTensor:
        bsz, N = parent_indices.shape
        
        n_parent_clusters, cumulative_n_child_clusters = torch.max(parent_indices).item() + 1, 0
        result = torch.zeros_like(parent_indices)
        for parent_cluster_idx in range(n_parent_clusters):
            index = parent_indices == parent_cluster_idx
            cluster_x = x[index]                                            # [n x embed_dim]
            
            ncut_x, eigenvalues = self.ncut.fit_transform(cluster_x)        # [n x ncut_dim], [ncut_dim]
            n_child_clusters = torch.sum(eigenvalues > self.config.eigenvalue_threshold)
            
            if n_child_clusters > 0:
                child_indices = torch.tensor(KMeans(n_clusters=n_child_clusters).fit_predict(ncut_x[:, :n_child_clusters]), dtype=torch.long)   # [n]
                result[index] = cumulative_n_child_clusters + child_indices
            else:
                result[index] = cumulative_n_child_clusters
            cumulative_n_child_clusters += n_child_clusters
            
            
            
            def visualize(ncut_x, normalized_ncut_x):
                hms = ncut_x.reshape(bsz, 28, 28, 2, self.config.ncut_dim // 2).permute(0, 3, 1, 4, 2).flatten(3, 4).flatten(1, 2)
                for nc, hm in [*zip(ncut_x, hms)][:3]:
                    plt.rcParams["figure.figsize"] = (5.0 * 2, 5.0 * (self.config.ncut_dim // 2))
                    plt.imshow(hm.numpy(force=True), cmap="bwr")
                    plt.title(f"Iteration {it}")
                    plt.axis("off")
                    plt.show()   
                
                all_labels = OrderedDict()
                # labels = torch.tensor(numpy_HDBSCAN(min_cluster_size=int(0.1 * bsz * N)).fit_predict(normalized_ncut_x.numpy(force=True)), dtype=torch.long).reshape(bsz, N)
                spectral_x, _ = self.iterative_ncut.fit_transform(normalized_ncut_x)
                all_labels["hdbscan"] = labels = torch.tensor(HDBSCAN(
                    min_cluster_size=int(self.config.cluster_size_threshold * bsz * N),
                    min_samples=512
                ).fit_predict(spectral_x), dtype=torch.long)
                
                n_child_clusters = torch.max(all_labels["hdbscan"]).item() + 1
                unclustered_indices = labels == -1
                
                print(f"n_child_clusters: {n_child_clusters}")
                if n_child_clusters == 0:
                    return
                
                spectral_x = spectral_x[:, :n_child_clusters]
                # KMeans Spectral
                # spectral_x, _ = NCUT(num_eig=n_child_clusters, affinity_focal_gamma=2.0, device=DEVICE).fit_transform(normalized_ncut_x)
                cluster_centers = torch.zeros((n_child_clusters, n_child_clusters))
                for cluster_idx in range(n_child_clusters):
                    cluster_centers[cluster_idx] = torch.mean(spectral_x[labels == cluster_idx], dim=0)
                # km_spectral_labels = torch.clone(labels)
                all_labels["km_boosted_spectral"] = torch.argmin(torch.cdist(spectral_x, cluster_centers), dim=1)
                all_labels["km_spectral"] = torch.tensor(KMeans(
                    n_clusters=n_child_clusters,
                    init=cluster_centers
                ).fit_predict(spectral_x), dtype=torch.long)
            
                # Axis-aligned Spectral
                aa_boosted_one_hot, RT = kway_ncut(spectral_x[~unclustered_indices], return_rotation=True)
                all_labels["aa_boosted_spectral"] = torch.argmax(spectral_x @ RT, dim=1)
                all_labels["aa_spectral"] = torch.argmax(kway_ncut(spectral_x, return_rotation=False), dim=1)
            
                all_labels = OrderedDict([
                    (k, v.reshape(bsz, N))
                    for k, v in all_labels.items()
                ])
                # print([torch.unique(labels_).tolist() for labels_ in all_labels])
                
                import einops
                import matplotlib as mpl
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                fig, axs = plt.subplots(nrows=len(all_labels), ncols=1)
                
                for j, (k, labels_) in enumerate(all_labels.items()):
                    cluster_im = torch.zeros((bsz, N, 3))
                    for child_cluster_idx in range(n_child_clusters):
                        cluster_im[labels_ == child_cluster_idx] = torch.tensor(mpl.colors.to_rgb(colors[child_cluster_idx]))
                        
                    num_ims = 3
                    axs[j].axis("off")
                    axs[j].imshow(einops.rearrange(
                        cluster_im[:num_ims], "bsz (h w) c -> h (bsz w) c",
                        h=28, w=28
                    ).numpy(force=True))
                    axs[j].set_title(k)
                plt.show()
     
    
            
            # """
            kwargs = {"dim": (0,), "keepdim": True}
            s = 1.0
            
            for it in range(2):
                print(f"iteration {it}")
                # normalized_ncut_x = ncut_x
                normalized_ncut_x = Fn.normalize(ncut_x, dim=-1)
                # normalized_ncut_x = (ncut_x - torch.mean(ncut_x, **kwargs)) / torch.std(ncut_x, **kwargs)
                # min_, max_ = torch.amin(ncut_x, **kwargs), torch.amax(ncut_x, **kwargs)
                # normalized_ncut_x = (2 * ncut_x - (max_ + min_)) / (max_ - min_)
                # normalized_ncut_x = torch.sigmoid(s * normalized_ncut_x)
                # normalized_ncut_x = (ncut_x - min_) / (max_ - min_)
                
                # [bsz x (N) x (ncut_dim)]
                print(eigenvalues)
                visualize(ncut_x, normalized_ncut_x)  
                
                ncut_x, eigenvalues = self.iterative_ncut.fit_transform(normalized_ncut_x)  
            
            # import seaborn as sns
            # plt.rcdefaults()
            # for i in range(self.config.ncut_dim):
            #     sns.kdeplot(normalized_ncut_x[:, i].numpy(force=True))
            #     # plt.hist(normalized_ncut_x[:, i].numpy(force=True), bins=100)
            #     plt.show()
            
            # ncut_x, eigenvalues = self.ncut.fit_transform(normalized_ncut_x)
            # print("Final RBF", eigenvalues)
            
            # normalized_ncut_x = ncut_x
            # visualize(ncut_x, normalized_ncut_x)
            
            
            raise Exception()
            
    
            """
            # min_cluster_size = 100
            plt.rcParams["figure.figsize"] = (7.0, 5.0)
            x_minmax = torch.tensor((50.0, 10000.0))
            x = torch.unique(torch.linspace(*x_minmax.log(), steps=30).exp().to(int)).tolist()
            y = []
            for min_cluster_size in x:
                labels = torch.tensor(HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(normalized_ncut_x.flatten(0, 1).numpy(force=True)))
                y.append((torch.max(labels) + 1).numpy(force=True))
            plt.plot(x, y)
            plt.xlabel("min_cluster_size")
            plt.xscale("log")
            plt.ylabel("n_clusters")
            plt.ylim(top=10)
            plt.show()   
            # print("unique:", torch.unique(labels))
            """
          
            
        print(result.shape, result.dtype, torch.unique(result))
            
        raise Exception()
            
        
        
        

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
            flattened_ncut_x, _ = self.ncut.fit_transform(flattened_x)                      # [(bsz * N) x N_D]
            ncut_x = flattened_ncut_x.reshape(bsz, N, self.config.ncut_dim)                        # [bsz x N x N_D]

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

                    child_centroid_initializations = torch.zeros((n_child_clusters, self.config.ncut_dim))
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
        # from sklearn.manifold import TSNE
        x_embedded_2d = TSNE(n_components=2).fit_transform(flattened_ncut_x.detach()).reshape(bsz, N, 2)
        x_embedded_3d = TSNE(n_components=3).fit_transform(flattened_ncut_x.detach()).reshape(bsz, N, 3)

        for i, (im_x_2d, im_x_3d) in enumerate(zip(x_embedded_2d, x_embedded_3d)):
            print(i)
            if i < 3:
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




