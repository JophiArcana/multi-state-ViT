import requests
from PIL import Image

import datasets
import torch
import torch.nn as nn
import torch_cluster
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT
from sklearn.manifold import TSNE

from transformers import ViTPreTrainedModel
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, ViTMAEModel
from transformers.image_processing_utils import BaseImageProcessor

from infrastructure.settings import *


if __name__ == "__main__":

    # M = nn.Parameter(torch.randn((1000, 5)))
    # # V, L = NCUT(num_eig=10, device=DEVICE).fit_transform(M)
    # # print(V.shape, L.shape)
    # # print(L, V[:, 0])
    # indices = torch_cluster.fps(M, batch=torch.arange(1000), ratio=0.25)
    # # print(indices, indices.shape)
    # raise Exception()

    dataset_name = "UCSC-VLAA/Recap-COCO-30K"
    model_name = "openai/clip-vit-base-patch32"

    # SECTION: Dataset setup
    dataset = datasets.load_dataset(dataset_name)
    dataset_size = dataset["train"].num_rows
    subsample_size = 10

    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    images = [
        dataset["train"][i]["image"]
        for i in range(subsample_size)
    ]
    inputs = image_processor(images=images, return_tensors="pt")

    # SECTION: Debugging
    from transformers import ViTModel
    from model.multistate_encoder.modeling_msvitencoder import MultiStateViTConfig, MultiStateViTEncoderModel
    from model.clustering import ClusteringConfig, NCutFPSClustering

    base = ViTModel.from_pretrained('facebook/dino-vitb8')

    model = MultiStateViTEncoderModel(MultiStateViTConfig(
        **base.config.to_dict(),
        generation_period=2,
        clustering_method="spectral",
        clustering_config=ClusteringConfig(
            ncut_dim=50, fps_dim=8, fps_ratio=0.5, nms_radius=0.1
        ),
    ))
    print(model)
    print(model.config.to_dict())
    # for image in images[:3]:
    #     plt.imshow(image)
    #     plt.show()
    print(model(**inputs))
    raise Exception()

    # SECTION: Model setup
    model = CLIPVisionModel.from_pretrained(model_name)
    # print(model)

    print(model)
    print(model.config)

    # model.embeddings(**inputs)
    a = model.get_input_embeddings()(inputs["pixel_values"])
    print(a.dtype, a.shape)
    import inspect

    print()
    print(model.vision_model.embeddings(inputs["pixel_values"]).shape)
    raise Exception()

    affinity_focal_gamma = 1.0
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for layer, states in enumerate(hidden_states):
            X = states[..., 1:, :].flatten(0, -2)

            normalized_X = torch.nn.functional.normalize(X, dim=-1)
            normalized_A = 1.0 - normalized_X @ normalized_X.mT
            A = (X.norm(dim=-1)[:, None] * X.norm(dim=-1)[None, :]) * normalized_A

            A = torch.exp(-A / affinity_focal_gamma)

            D = A.sum(dim=-1)
            L = torch.eye(len(D)) - A * ((D[:, None] * D[None, :]) ** -0.5)

            E, V = torch.linalg.eigh(L)
            X = V[:, :10]

            X_embedded = TSNE(n_components=2).fit_transform(X)

            plt.scatter(*X_embedded.T)
            plt.title(f"Layer {layer}")
            plt.show()

            print(layer, states.shape, X_embedded.shape)






