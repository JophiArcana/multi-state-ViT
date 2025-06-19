from typing import Dict, Literal

from .modeling_predictor import AbstractSaccadicViTPredictor
from .modeling_transformer_predictor import TransformerSaccadicViTPredictor as _TransformerSaccadicViTPredictor


RefinerImplementationOptions = Literal["linear", "transformer"]
SACCADIC_VIT_PREDICTOR_CLASSES: Dict[RefinerImplementationOptions, type] = {
    "linear": NotImplementedError(),
    "transformer": _TransformerSaccadicViTPredictor,
}
