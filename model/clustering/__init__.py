from typing import Dict

from model.clustering.modeling_fps import FPSClustering
from model.clustering.modeling_hdbscan import HDBBoostedSpectralClustering


CLUSTERING_CLASSES: Dict[str, type] = {
    "fps": FPSClustering,
    "spectral": HDBBoostedSpectralClustering,
}




