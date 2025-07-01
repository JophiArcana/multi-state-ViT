#%%
import sys
sys.path.append("/workspace/multi-state-ViT")
from infrastructure.settings import DEVICE

import datasets
import numpy as np
import torch
from matplotlib import pyplot as plt

from infrastructure.dataset import DATASETS


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # torch.manual_seed(11163065284731897840)
    torch.manual_seed(1212)
    # torch.manual_seed(10936251301023808035)
    # print(f"Seed: {torch.seed()}")
    
    dataset_name, n_classes = DATASETS["Common"][0]
    base_model_name = "facebook/dino-vitb8"

    # SECTION: Dataset setup
    dataset = datasets.load_dataset(dataset_name)
    dataset_size = dataset["train"].num_rows

    images = []
    class_idx, subsample_size = {0, 1, 2}, 50
    while len(images) < subsample_size:
        image = dataset["train"][torch.randint(0, dataset_size, ()).item()]
        if image["label"] in class_idx:
            images.append(image["image"])

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
    images = [*map(process_grayscale, images)]

    # SECTION: Debugging
    from transformers import ViTModel, ViTImageProcessor
    from model.multistate_encoder.modeling_msvitencoder import MultiStateViTConfig, MultiStateViTEncoderModel
    from model.clustering.modeling_spectral import SpectralClusteringConfig

    base = ViTModel.from_pretrained(base_model_name)

    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
    inputs = image_processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)

    model = MultiStateViTEncoderModel(MultiStateViTConfig(
        **base.config.to_dict(),
        _attn_implementation="eager",
        pregeneration_period=10,
        generation_period=2,
        clustering_config=SpectralClusteringConfig(
            ncut_dim=8,
            ncut_dist="rbf",
            eigenvalue_threshold=0.1,
            cluster_size_threshold=0.07,
        ),
        # clustering_config=FPSClusteringConfig(
        #     ncut_dim=100,
        #     fps_dim=8,
        #     fps_sample1=300,
        #     fps_sample2=100,
        #     fps_supersample2=120,
        #     cosine_similarity_threshold=0.7,
        # ),
        pretrained=base_model_name
    ))
    print(model)
    print(model.config)
    for image in images[:3]:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    with torch.no_grad():
        print(model(inputs, interpolate_pos_encoding=True))
    raise Exception()

    # SECTION: Model setup
    model = CLIPVisionModel.from_pretrained(model_name)
    # print(model)

    print(model)
    print(model.config)

    # model.embeddings(**inputs)
    a = model.get_input_embeddings()(inputs["pixel_values"])
    print(a.dtype, a.shape)

    print()
    print(model.vision_model.patch_embed(inputs["pixel_values"]).shape)
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





# %%
