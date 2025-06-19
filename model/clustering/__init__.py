from typing import Dict

from model.clustering.modeling_fps import FPSClustering
from model.clustering.modeling_spectral import SpectralClustering


CLUSTERING_CLASSES: Dict[str, type] = {
    "fps": FPSClustering,
    "spectral": SpectralClustering,
}




