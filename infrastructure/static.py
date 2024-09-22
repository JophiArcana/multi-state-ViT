from typing import *

import numpy as np
from tensordict import TensorDict

from system.base import SystemGroup

PARAM_GROUP_FORMATTER: str = "{0}_d({1})"
TRAINING_DATASET_TYPES: List[str] = [
    "train",
    "valid",
]
TESTING_DATASET_TYPE: str = "test"
DATASET_SUPPORT_PARAMS: List[str] = [
    "dataset_size",
    "total_sequence_length",
    "system.n_systems"
]
INFO_DTYPE: np.dtype = np.dtype([
    ("systems", SystemGroup),
    ("system_params", object),
    ("dataset", object)
])
RESULT_DTYPE: np.dtype = np.dtype([
    ("time", float),
    ("output", TensorDict),
    ("learned_kfs", tuple),
    ("systems", object),
    ("metrics", object),
])




