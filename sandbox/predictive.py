import datasets
import numpy as np
import torch
from matplotlib import pyplot as plt

from infrastructure.dataset import DATASETS
from infrastructure.settings import DEVICE


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # torch.manual_seed(11163065284731897840)
    # torch.manual_seed(1212)
    # torch.manual_seed(10936251301023808035)
    # print(f"Seed: {torch.seed()}")

    dataset_name, n_classes = DATASETS["Common"][0]
    base_model_name = "facebook/dino-vitb8"

    # SECTION: Dataset setup
    dataset = datasets.load_dataset(dataset_name)
    dataset_size = dataset["train"].num_rows

    images = []
    subsample_size = 5
    while len(images) < subsample_size:
        images.append(dataset["train"][torch.randint(0, dataset_size, ()).item()]["image"])

    def process_grayscale(im):
        arr = np.array(im)
        return arr if arr.ndim >= 3 else np.tile(arr[..., None], (1, 1, 3))
    images = [*map(process_grayscale, images)]

    # SECTION: Configure model
    from transformers import ViTModel, ViTImageProcessor
    from model.predictive_encoder.modeling_spvitencoder import PredictiveViTConfig, PredictiveViTModel

    base_model = ViTModel.from_pretrained(base_model_name)

    image_size = 224
    image_processor = ViTImageProcessor.from_pretrained(base_model_name)
    image_processor.__dict__.update({
        "size": {"height": image_size, "width": image_size},
    })
    inputs = image_processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)

    model = PredictiveViTModel(PredictiveViTConfig(
        _attn_implementation="eager",
        patch_config_scale=[
            [0.7, 0.0],
            [0.7, 0.0],
            [0.5, -1.0],
        ],
        pretrained=base_model_name,
        patch_size=64,
    ))
    print(model)
    # print(model.config)
    # for image in images:
    #     plt.imshow(image)
    #     plt.axis("off")
    #     plt.show()

    with torch.no_grad():
        print(model.forward(pixel_values=inputs, output_attentions=True, output_hidden_states=True))




