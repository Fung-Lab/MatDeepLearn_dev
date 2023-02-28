import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    compute_bond_angles,
    custom_edge_feats,
    custom_node_feats,
    generate_virtual_nodes,
    get_cutoff_distance_matrix,
    get_mask,
)

from abc import ABC, abstractmethod


class Routine(ABC):
    """Abstract base class for routines, which are configurable and
    differentiable sets of operations designed to be used at both
    processing and model runtime stages.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, data: Data):
        pass
