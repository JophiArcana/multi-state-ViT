import requests
from PIL import Image

import datasets

from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, Dinov2Model

from infrastructure.dataset import DATASETS


if __name__ == "__main__":
    dataset_name, n_classes = DATASETS["Common"][1]
    print(dataset_name)
    dataset = datasets.load_dataset(dataset_name)
    print(dataset.shape)

    base_model_name = "openai/clip-vit-large-patch14"
    model = CLIPVisionModel.from_pretrained(base_model_name)
    model.eval()

    images = [d["image"] for d in [*dataset["train"]][:100]]
    image_processor = CLIPImageProcessor.from_pretrained(base_model_name)
    raise Exception()

    features = []
    with torch.no_grad():
        inputs = image_processor(images=images, return_tensors="pt")
        print(model(**inputs).pooler_output.shape)





    # FID = FrechetInceptionDistance()
    # FID.update([
    #     dataset["train"][i]["image"]
    #     for i in range(5)
    # ], True)
    # # FID.update()




