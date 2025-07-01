import torch
import datasets


if __name__ == "__main__":
    ds = datasets.load_dataset("dalle-mini/open-images", split="train", trust_remote_code=True)
    print(ds)











