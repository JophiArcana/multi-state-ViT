import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..subsample_encoder import ImageClassifierOutputWithLog


def reverse_normalize_im(im):
    return (im * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]


def visualize_subsample_vit_output(
    output: ImageClassifierOutputWithLog,
    num_ims: int = 3,
) -> None:
    max_depth = len(output.valid_masks) - 1
    nrows, ncols = 2, (max_depth + 1) * num_ims
    plt.rcParams["figure.figsize"] = (2.0 * ncols, 2.0 * nrows,)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for im_idx in range(num_ims):
        for depth in range(max_depth + 1):

            valid_mask = output.valid_masks[depth][im_idx]
            corners = output.corners[depth][im_idx][valid_mask].numpy(force=True)

            # Plot nested image
            pixel_values = np.clip(reverse_normalize_im(einops.rearrange(
                output.pixel_values[depth][im_idx][valid_mask],
                "... c h w -> ... h w c",
            ).numpy(force=True)), a_min=0.0, a_max=1.0)

            ax_im: Axes = axs[0, im_idx * (max_depth + 1) + depth]
            ax_im.axis("off")
            ax_im.set_aspect("equal")
            for pv, c in zip(pixel_values, corners):
                ax_im.imshow(pv, extent=(c[0, 1], c[1, 1], c[1, 0], c[0, 0]), interpolation="none",)
            for c in corners:
                ax_im.plot(
                    [c[0, 1], c[0, 1], c[1, 1], c[1, 1], c[0, 1],],
                    [c[0, 0], c[1, 0], c[1, 0], c[0, 0], c[0, 0],],
                    color="gold", linewidth=2.0 * (c[1, 0] - c[0, 0]), linestyle="--",
                )
            ax_im.set_title(f"Image {im_idx}: depth {depth}")

            # Plot nested decision tree
            sigmoid = torch.sigmoid(output.subsample_logits[depth][im_idx][valid_mask]).numpy(force=True)
            vmin, vmax = 0.4, 0.6   # min(sigmoid), max(sigmoid)

            ax_dt: Axes = axs[1, im_idx * (max_depth + 1) + depth]
            ax_dt.axis("off")
            ax_dt.set_aspect("equal")
            for s, c in zip(sigmoid, corners):
                im = ax_dt.imshow(
                    s[None, None],
                    cmap="seismic", vmin=vmin, vmax=vmax, extent=(c[0, 1], c[1, 1], c[1, 0], c[0, 0]), interpolation="none",
                )

                center = (c[0] + c[1]) / 2
                ax_dt.text(
                    center[1], center[0], f"{s:.2f}",
                    fontsize=12.0 * (c[1, 0] - c[0, 0]) ** 0.5, ha="center", va="center",
                )

            for c in corners:
                ax_dt.plot(
                    [c[0, 1], c[0, 1], c[1, 1], c[1, 1], c[0, 1],],
                    [c[0, 0], c[1, 0], c[1, 0], c[0, 0], c[0, 0],],
                    color="black", linewidth=0.2 * (c[1, 0] - c[0, 0]), alpha=0.5,
                )
            fig.colorbar(im, cax=make_axes_locatable(ax_dt).append_axes("right", size="5%", pad=0.05), orientation="vertical")

    plt.show()
    plt.close()
    plt.rcdefaults()