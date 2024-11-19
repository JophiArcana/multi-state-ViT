import torch
import torch.nn as nn
import torch.nn.functional as Fn
from fast_pytorch_kmeans import KMeans
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT
from sklearn.cluster import SpectralClustering

from infrastructure.settings import *


if __name__ == "__main__":
    torch.manual_seed(1212)
    torch.set_printoptions(precision=6, sci_mode=False, linewidth=400)

    M = torch.randn((8, 2))

    torch.manual_seed(2003)
    NC = NCUT(num_eig=2, distance="rbf", normalize_features=False)
    X, L = NC.fit_transform(M)
    print(X, L)

    # torch.manual_seed(2003)
    # NC = NCUT(num_eig=2, distance="cosine", normalize_features=False)
    # M_ = Fn.normalize(M, dim=-1)
    # X_, L_ = NC.fit_transform(M_)
    # print(X_, L_)

    # print(X == X_, L == L_)
    raise Exception()



    for it in range(1000):
        N = NCUT(num_eig=n_eigs, normalize_features=False, distance="rbf")
        f = N.fit_transform(x.flatten(0, 1))[0].unflatten(0, (n_clusters, n_points))
        f = (f - f.mean(dim=[0, 1])) / f.std(dim=[0, 1])

        temp = 1 * (it + 1)
        # sm = torch.nn.functional.softmax(f, dim=-1)
        sm = torch.nn.functional.gumbel_softmax(f * temp, dim=-1, tau=temp)
        m = torch.nn.functional.one_hot(torch.argmax(sm, dim=-1), num_classes=n_eigs).to(torch.float).detach()

        loss = torch.nn.functional.binary_cross_entropy(
            sm.flatten(0, 1) @ sm.flatten(0, 1).mT,
            m.flatten(0, 1) @ m.flatten(0, 1).mT, reduce=False
        ).mean()
        # loss = torch.nn.functional.cross_entropy(sm.log().flatten(0, 1), m.flatten(0, 1))

        match = sm.argmax(dim=-1) == f.argmax(dim=-1)
        print(f"Loss: {loss.item()}, stay proportion: {match.sum() / match.numel()}, cross proportion: {1 - match.sum() / match.numel()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # rescale()

        if it % 1 == 0:
            with torch.no_grad():
                f_indices = torch.argmax(f, dim=-1)
                fig, axs = plt.subplots(ncols=n_eigs + 1)
                for i in range(n_eigs):
                    axs[i].scatter(*x.flatten(0, 1).mT, c=f[..., i].flatten(0, 1), cmap="bwr", s=0.5, label="positive")
                    axs[i].set_title(f"eigenvalue{i}")
                    axs[n_eigs].scatter(*x[f_indices == i].mT, s=0.5)
                axs[n_eigs].set_title(f"Iteration {it}")
                plt.show()





