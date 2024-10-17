"""MultiStateViT model configuration"""

from typing import Any

from transformers import ViTConfig

from model.clustering.modeling import ClusteringConfig


class MultiStateViTConfig(ViTConfig):
    r"""
    This is the configuration class to store the configuration of a [`MultiStateViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        generation_period (`int`, *optional*, defaults to 2):
            Number of transformer layers between hierarchical subclusterings.
        clustering_method (`str`, *optional*, defaults to `"spectral"`):
            Clustering method used for hierarchical subclustering in cluster-restricted attention.
    ```"""
    def __init__(
        self,
        pregeneration_period: int = 4,
        generation_period: int = 2,
        attention_mask_inf: float = 1e2,
        clustering_config: ClusteringConfig = None,
        pretrained: str = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.pregeneration_period = pregeneration_period
        self.generation_period = generation_period
        self.attention_mask_inf = attention_mask_inf
        self.clustering_config = clustering_config
        self.pretrained = pretrained




