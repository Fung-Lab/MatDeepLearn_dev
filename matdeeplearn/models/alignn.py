from typing import Literal, Optional

import numpy as np
import torch
from torch.nn import BatchNorm1d, Linear, Sequential, Sigmoid, SiLU
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


@registry.register_model("ALIGNN")
class ALIGNN(BaseModel):
    def __init__(
        self,
        data,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        atom_input_features: int = 114,
        edge_input_features: int = 50,
        triplet_input_features: int = 40,
        embedding_features: int = 64,
        hidden_features: int = 256,
        output_features: int = 1,
        min_edge_distance: float = 0.0,
        max_edge_distance: float = 8.0,
        min_angle: float = 0.0,
        max_angle: float = torch.acos(torch.zeros(1)).item() * 2,
        link: Literal["identity", "log", "logit"] = "identity",
        **kwargs,
    ) -> None:
        super().__init__()

        # utilizing data object
        atom_input_features = data.num_features
        edge_input_features = data.num_edge_features

        self.atom_embedding = EmbeddingLayer(atom_input_features, hidden_features)

        self.edge_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=min_edge_distance, vmax=max_edge_distance, bins=edge_input_features
            ),
            EmbeddingLayer(edge_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features),
        )

        self.angle_embedding = torch.nn.Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=triplet_input_features),
            EmbeddingLayer(triplet_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features),
        )

        # layer to perform M ALIGNNConv updates on the graph
        self.alignn_layers = torch.nn.ModuleList(
            [ALIGNNConv(hidden_features, hidden_features) for _ in range(alignn_layers)]
        )

        # layer to perform N EdgeGatedConv updates on the graph
        self.gcn_layers = torch.nn.ModuleList(
            [
                EdgeGatedGraphConv(hidden_features, hidden_features)
                for _ in range(gcn_layers)
            ]
        )

        # prediction task
        self.fc = Linear(hidden_features, output_features)

        # linking which is performed post-readout
        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
        elif link == "logit":
            self.link = torch.sigmoid

    @property
    def target_attr(self):
        return "y"

    def forward(self, g: Data):
        # initial node features
        g.x = g.x.to(dtype=torch.float)
        node_feats = self.atom_embedding(g.x)
        # initial bond features
        edge_attr = self.edge_embedding(g.edge_attr)
        # initial angle/triplet features
        triplet_feats = self.angle_embedding(g.edge_attr_lg)

        # ALIGNN updates
        for alignn_layer in self.alignn_layers:
            node_feats, edge_attr, triplet_feats = alignn_layer(
                g,  # required for correct edge and triplet indexing
                node_feats,
                edge_attr,
                triplet_feats,
            )

        # GCN updates
        for gcn_layer in self.gcn_layers:
            node_feats, edge_attr = gcn_layer(
                node_feats,
                edge_attr,
                g.edge_index,
            )

        # readout
        h = global_mean_pool(node_feats, g.batch)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out, -1)


class ALIGNNConv(torch.nn.Module):
    """
    Implementation of the ALIGNN layer composed of EdgeGatedGraphConv steps
    """

    def __init__(self, input_features, output_features) -> None:
        super().__init__()

        # Sequential EdgeGatedCONV layers
        # Overall mapping is input_features -> output_features
        self.edge_update = EdgeGatedGraphConv(output_features, output_features)
        self.node_update = EdgeGatedGraphConv(input_features, output_features)

    def forward(
        self,
        g: Data,
        node_feats: torch.Tensor,
        edge_attr: torch.Tensor,
        triplet_feats: torch.Tensor,
    ) -> torch.Tensor:
        # Perform sequential edge and node updates

        message, triplet_feats = self.edge_update(
            edge_attr, triplet_feats, g.edge_index_lg
        )

        node_feats, edge_attr = self.node_update(node_feats, message, g.edge_index)

        # Return updated node, edge, and triplet embeddings
        return node_feats, edge_attr, triplet_feats


class EdgeGatedGraphConv(MessagePassing):
    """
    Message-passing based implementation of EGGConv
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True, eps=1e-6
    ) -> None:
        super().__init__()

        self.W_src = Linear(input_features, output_features)
        self.W_dst = Linear(input_features, output_features)
        # Operates on h_i
        self.W_ai = Linear(input_features, output_features)
        # Operates on h_j
        self.W_bj = Linear(input_features, output_features)
        # Operates on e_ij
        self.W_cij = Linear(output_features, output_features)

        self.bn_nodes = BatchNorm1d(output_features)
        self.bn_edges = BatchNorm1d(output_features)

        self.act = SiLU()
        self.sigmoid = Sigmoid()
        self.residual = residual
        self.eps = eps

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        i, j = edge_index
        # Node update routine
        sigma = self.sigmoid(edge_attr)
        sigma_sum = scatter(src=sigma, index=i, dim=0)
        # Accessing at index i allows for shape matching and correct aggregate division
        e_ij_hat = sigma / (sigma_sum[i] + self.eps)

        dest_aggr = self.propagate(edge_index, x=node_feats, e_ij_hat=e_ij_hat)

        new_node_feats = node_feats + self.act(
            self.bn_nodes(self.W_src(node_feats) + dest_aggr)
        )

        # Edge update routine
        new_edge_attr = edge_attr + self.act(
            self.bn_edges(
                self.W_ai(node_feats[i])
                + self.W_bj(node_feats[j])
                + self.W_cij(edge_attr)
            )
        )

        return new_node_feats, new_edge_attr

    def message(self, x_j, e_ij_hat):
        return e_ij_hat * self.W_dst(x_j)


class EdgeGatedGraphConvNoMP(torch.nn.Module):
    """
    Implementation of the EdgeGatedGraphConv layer
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ) -> None:
        super().__init__()

        # define the edge and node models for creating new embeddings
        self.edge_model = EdgeModel(input_features, output_features, residual)
        self.node_model = NodeModel(input_features, output_features, residual)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # compute new edge features

        row, col = edge_index
        new_edge_attr = self.edge_model(node_feats[row], node_feats[col], edge_attr)

        # compute new node features (which are based on previously updated edge features)
        new_node_feats = self.node_model(node_feats, edge_index, new_edge_attr)

        return new_node_feats, new_edge_attr


class EdgeModel(torch.nn.Module):
    """
    Abstraction to perform an update on the edge attributes
    e_ij_new = f(e_ij, h_i, h_j)
    """

    def __init__(self, input_features, output_features, residual=True):
        super().__init__()

        # source node attrributes
        self.src_gate = Linear(input_features, output_features)
        # dest node attributes
        self.dest_gate = Linear(input_features, output_features)
        # weights for edge attributes
        self.edge_gate = Linear(input_features, output_features)

        self.batch_norm = BatchNorm1d(output_features)
        self.residual = residual

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # src and dest are the nodes connecting the edge

        new_feats = F.silu(
            self.batch_norm(
                self.src_gate(src) + self.dest_gate(dest) + self.edge_gate(edge_attr)
            )
        )

        if self.residual:
            new_feats = edge_attr + new_feats
        return new_feats


class NodeModel(torch.nn.Module):
    """
    Abstraction to perform an update on the node attributes
    h_i_new = f(h_i, \\sum e_ij_hat * (Wdst * h_j))
    """

    def __init__(
        self, input_features, output_features, residual=True, eps=1e-6
    ) -> None:
        super().__init__()
        # sam
        self.src_update = Linear(input_features, output_features)
        # Define message passing routines
        self.node_aggr = NodeAggregation(input_features, output_features)
        self.edge_aggr = EdgeAggregation()

        self.batch_norm = BatchNorm1d(output_features)
        self.act = SiLU()
        self.sigmoid = Sigmoid()

        self.residual = residual
        self.eps = eps

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # compute sigmoid-aggregate of new edge features
        node_aggregate = self.node_aggr(x, edge_index, edge_attr)

        edge_aggregate = self.edge_aggr(edge_index, edge_attr)

        dest_aggr = node_aggregate / (edge_aggregate + self.eps)

        # compute new node features
        new_feats = self.act(self.batch_norm(self.src_update(x) + dest_aggr))

        if self.residual:
            new_feats = x + new_feats

        return new_feats


class NodeAggregation(MessagePassing):
    """
    Used to compute an aggregation of transformed node attributes and edge attributes
    \\sumj e_ij_hat * (Wdst * h_j)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        # Define the MLP that operates on each neighboring node
        self.dst_update = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        update = self.dst_update(x_j)
        # element-wise multiplication as dest update matches shapes
        return torch.sigmoid(edge_attr) * update


class EdgeAggregation(MessagePassing):
    """
    Used to compute the aggregation of edge attributes (sigmoid transform) with respect to neighboring nodes
    Message passing still occurs with respect to bond graph
    \\sumk \\sigma(e_ik)
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, edge_index, edge_attr):
        out = self.propagate(edge_index, edge_attr=edge_attr)
        # print(out.element_size() * out.nelement(), out.nelement())
        return out

    def message(self, edge_attr):
        return torch.sigmoid(edge_attr)


class EmbeddingLayer(torch.nn.Module):
    """
    Custom layer which performs nonlinear transform on embeddings
    """

    def __init__(self, input_features, output_features) -> None:
        super().__init__()

        self.mlp = Sequential(
            Linear(input_features, output_features),
            BatchNorm1d(output_features),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RBFExpansion(torch.nn.Module):
    """
    RBF Expansion on distances or angles to compute Gaussian distribution of embeddings
    """

    def __init__(
        self,
        vmin: float = 0,  # default 0A
        vmax: float = 8,  # default 8A
        bins: int = 40,  # embedding dimension
        lengthscale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)

        if lengthscale is None:
            lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 / lengthscale
            self.lengthscale = lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale**2)

    def forward(self, distance: torch.Tensor):
        out = torch.exp(
            -self.gamma * (distance - self.centers.to(distance.device)) ** 2
        )
        return out
