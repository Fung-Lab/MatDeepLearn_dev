from typing import List

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import CGConv, GATConv, Set2Set

import matdeeplearn.models.routines.pooling as pooling
from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from collections import defaultdict


class SemanticAttention(nn.Module):
    """
    Semantic-level attention.
    Adapted from https://github.com/Jhy1993/HAN/blob/master/utils/layers.py
    """

    def __init__(self, meta_paths: list[str], embed_dim: int) -> None:
        super().__init__()
        self.meta_paths = meta_paths
        # weights and attention vectors for each meta path
        self.w_dict = nn.ModuleDict(
            {mp: nn.Linear(embed_dim, embed_dim) for mp in meta_paths}
        )
        self.q_dict = nn.ParameterDict(
            {mp: nn.Parameter(torch.ones((embed_dim, 1))) for mp in meta_paths}
        )

    def forward(self, path_embds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        meta_path_attns = defaultdict(nn.Module)
        # compute attention scores and weights for each meta path
        for mp in self.meta_paths:
            x = path_embds[mp]
            # project embedding via linear layer
            projected_embed = self.w_dict[mp](x)
            # compute attention scores and weights
            scores = torch.tensordot(
                torch.tanh(projected_embed), self.q_dict[mp], dims=1
            ) / x.size(1)
            attn_weights = F.softmax(scores, dim=-1)
            meta_path_attns[mp] = attn_weights

        return meta_path_attns


@registry.register_model("CGCNN_HAN_VN")
class CGCNN_HAN_VN(BaseModel):
    def __init__(
        self,
        edge_steps,
        self_loop,
        data,
        dim1=100,
        dim2=150,
        pre_fc_count=1,
        gc_count=4,
        post_fc_count=3,
        attn_heads=1,
        heterogeneous_conv=False,
        pool="global_mean_pool",
        virtual_pool="AtomicNumberPooling",
        pool_choice="both",
        mp_pattern: List[str] = ["rv", "rr"],
        atomic_intermediate_layer_resolution=0,
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act_fn="relu",
        act_nn="ReLU",
        dropout_rate=0.0,
    ):
        super(CGCNN_HAN_VN, self).__init__(edge_steps, self_loop)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        self.virtual_pool = virtual_pool
        self.pool_choice = pool_choice
        self.mp_pattern = mp_pattern
        self.atomic_intermediate_layer_resolution = atomic_intermediate_layer_resolution
        self.act_fn = act_fn
        self.act_nn = act_nn
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.gc_count = gc_count
        self.post_fc_count = post_fc_count
        self.attn_heads = attn_heads
        self.heterogeneous_conv = heterogeneous_conv

        # Relying on data object attributes (data.num_edge_features) not recommended
        self.num_features = data.num_node_features
        self.num_edge_features = edge_steps

        assert len(self.mp_pattern) > 1, "Need at least 1 message passing routine."

        # Determine gc dimension and post_fc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.num_features, self.num_features
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        # Determine output dimension length
        if data[0][self.target_attr].ndim == 0:
            self.output_dim = 1
        else:
            self.output_dim = len(data[0][self.target_attr])

        # setup layers
        self.pre_lin_virtual, self.pre_lin_real = (
            self._setup_pre_gnn_layers(),
            self._setup_pre_gnn_layers(),
        )
        # heterogeneous conv performs different convolutions for each MP interaction
        if self.heterogeneous_conv:
            gnn_list, bn_list = self._setup_gnn_layers()
            for p in self.mp_pattern:
                setattr(self, f"conv_{p}_list", gnn_list)
                setattr(self, f"bn_{p}_list", bn_list)
        else:
            self.conv_list, self.bn_list = self._setup_gnn_layers()

        # node-level attention
        self.attn_conv_list = self._setup_node_attn_layers()
        # semantic attention
        self.semantic_attn = SemanticAttention(self.mp_pattern, self.dim1)
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()

        # Should processing_steps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(self.post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not recommended to use set2set
            self.lin_out_2 = torch.nn.Linear(self.output_dim * 2, self.output_dim)

        # virtual node pooling scheme
        self.virtual_node_pool = (
            getattr(pooling, self.virtual_pool)(
                self.pool,
                pool_choice=self.pool_choice,
            )
            if self.virtual_pool != ""
            else None
        )

    @property
    def target_attr(self):
        return "y"

    def _setup_pre_gnn_layers(self) -> torch.nn.ModuleList:
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()

        if self.pre_fc_count > 0:
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin_r = torch.nn.Linear(self.num_features, self.dim1)
                else:
                    lin_r = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin_r)

        return pre_lin_list

    def _setup_node_attn_layers(self):
        """Set up node level attention layers."""
        attn_conv_list = torch.nn.ModuleList()

        for _ in range(len(self.mp_pattern)):
            attn_conv_list.append(
                GATConv(
                    self.gc_dim,
                    self.gc_dim,
                    heads=self.attn_heads,
                    concat=False,
                    dropout=self.dropout_rate,
                    bias=True,
                )
            )

        return attn_conv_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for _ in range(self.gc_count):
            conv = CGConv(
                self.gc_dim, self.num_edge_features, aggr="mean", batch_norm=False
            )
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats
                )
                bn_list.append(bn)

        return conv_list, bn_list

    def _setup_post_atomic_pooling_layers(self, post_fc: bool):
        if self.atomic_intermediate_layer_resolution > 0:
            # We use 100 as embedding expansion in atomic num pooling, can prevent loss of info with sequential layers
            interm_layers: List[torch.nn.Linear] = []

            # Find closest lesser multiple of self.dim2 to the input dim
            in_dim = self.post_fc_dim * 100 - (self.post_fc_dim * 100) % self.dim2
            # Find the resolution of the atomic number embedding, how many layers to add
            scale_factor = self.dim2 * self.atomic_intermediate_layer_resolution
            resolution = in_dim // scale_factor

            stride_dims = [scale_factor * i for i in range(resolution - 1, 0, -1)]

            for out_dim in stride_dims:
                interm_layers.append(torch.nn.Linear(in_dim, out_dim))
                torch.nn.Dropout(p=self.dropout_rate)
                in_dim = out_dim

            interm_layers.append(torch.nn.Linear(in_dim, self.dim2))

            layers = [
                torch.nn.Linear(self.post_fc_dim * 100, interm_layers[0].in_features),
                torch.nn.Dropout(p=self.dropout_rate),
                *interm_layers,
            ]
        else:
            layers = [
                torch.nn.Linear(
                    self.post_fc_dim * 100, self.dim2 if post_fc else self.output_dim
                ),
            ]
        return torch.nn.Sequential(*layers)

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number.
        In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                lin: torch.nn.Module
                if i == 0:
                    # Set2set pooling AND RV_node pooling has doubled dimension
                    if self.pool_order == "early" and (
                        self.pool == "set2set"
                        or self.virtual_pool == "RealVirtualPooling"
                    ):
                        if self.pool_choice == "both":
                            lin = torch.nn.Linear(self.post_fc_dim * 2, self.dim2)
                        elif (
                            self.pool_choice == "real" or self.pool_choice == "virtual"
                        ):
                            lin = torch.nn.Linear(self.post_fc_dim, self.dim2)
                    elif (
                        self.pool_order == "early"
                        and self.virtual_pool == "AtomicNumberPooling"
                    ):
                        # We use 100 as embedding expansion in atomic num pooling
                        lin = self._setup_post_atomic_pooling_layers(post_fc=True)
                    else:
                        lin = torch.nn.Linear(self.post_fc_dim, self.dim2)
                else:
                    lin = torch.nn.Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = torch.nn.Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and (
                self.pool == "set2set" or self.virtual_pool == "RealVirtualPooling"
            ):
                lin_out = torch.nn.Linear(self.post_fc_dim * 2, self.output_dim)
            elif (
                self.pool_order == "early"
                and self.virtual_pool == "AtomicNumberPooling"
            ):
                lin_out = self._setup_post_atomic_pooling_layers(post_fc=False)
            else:
                lin_out = torch.nn.Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out

    def forward(self, data: Data):
        # compute masked node feature matrices for real and virtual nodes
        virtual_node_mask = (
            (data.z == 100).type(torch.uint8).unsqueeze(1).repeat(1, data.x.shape[-1])
        )
        real_node_mask = (
            (data.z != 100).type(torch.uint8).unsqueeze(1).repeat(1, data.x.shape[-1])
        )
        # create separate embedding spaces for real and virtual nodes
        virtual_features = torch.zeros_like(data.x)
        real_features = torch.zeros_like(data.x)

        virtual_features = data.x * virtual_node_mask
        real_features = data.x * real_node_mask

        # pass each node feature matrix through a separate projection scheme
        for j in range(0, len(self.pre_lin_virtual)):
            if j == 0:
                out_v = self.pre_lin_virtual[j](virtual_features.float())
            else:
                out_v = self.pre_lin_virtual[j](out_v)
            out_v = getattr(F, self.act_fn)(out_v)

        for j in range(0, len(self.pre_lin_real)):
            if j == 0:
                out_r = self.pre_lin_real[j](real_features.float())
            else:
                out_r = self.pre_lin_real[j](out_r)
            out_r = getattr(F, self.act_fn)(out_r)

        # ignore bias computation
        out_v = out_v * virtual_node_mask
        out_r = out_r * real_node_mask

        # node level attention computations
        embedding_dict = {}
        for j, mp in enumerate(self.mp_pattern):
            edge_idx = getattr(data, f"edge_index_{mp}")
            if mp == "rr":
                embedding_dict[mp] = (
                    self.attn_conv_list[j](out_r, edge_idx) * real_node_mask
                )
            elif mp == "rv" or mp == "vr":
                embedding_dict[mp] = self.attn_conv_list[j](out_r + out_v, edge_idx)
            elif mp == "vv":
                embedding_dict[mp] = (
                    self.attn_conv_list[j](out_v, edge_idx) * virtual_node_mask
                )

        # semantic (meta-path) level attention
        attn_weights = self.semantic_attn(embedding_dict)

        if self.heterogeneous_conv:
            out = torch.zeros_like(data.x)
            for mp in self.mp_pattern:
                conv_list = getattr(self, f"conv_{mp}_list")
                bn_list = getattr(self, f"bn_{mp}_list")
                for j in range(len(conv_list)):
                    conv = conv_list[j]
                    edge_idx = getattr(data, f"edge_index_{mp}")
                    edge_attr = getattr(data, f"edge_attr_{mp}")
                    emb = conv(embedding_dict[mp], edge_idx, edge_attr)
                    if self.batch_norm:
                        bn = bn_list[j]
                        emb = bn(emb)
                # aggregate embeddings using attention weights
                out = out + emb * attn_weights[mp].expand(-1, emb.size(-1))
        else:
            # use the correct edge_indexes and edge_attrs for MP
            edge_index_use = torch.cat(
                [getattr(data, f"edge_index_{mp}") for mp in self.mp_pattern], dim=1
            )
            edge_attr_use = torch.cat(
                [getattr(data, f"edge_attr_{mp}") for mp in self.mp_pattern], dim=0
            )
            # obtain final output embedding using attention weights
            embeds = torch.stack([embedding_dict[mp] for mp in self.mp_pattern])
            attns = torch.stack([attn_weights[mp] for mp in self.mp_pattern])
            out = torch.sum(embeds * attns.expand(-1, -1, embeds.size(-1)), dim=0)
            # conv operations
            for j in range(0, len(self.conv_list)):
                if self.batch_norm:
                    out = self.conv_list[j](
                        out,
                        edge_index_use,
                        edge_attr_use.float(),
                    )
                    out = self.bn_list[j](out)
                else:
                    out = self.conv_list[j](
                        out,
                        edge_index_use,
                        edge_attr_use.float(),
                    )
                out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Post-GNN dense layers
        if self.pool_order == "early":
            # virtual node pooling scheme if chosen
            if self.virtual_node_pool is not None:
                out = self.virtual_node_pool(
                    data,
                    out,
                )
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

            for j in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[j](out)
                out = getattr(F, self.act_fn)(out)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            raise NotImplementedError("Late pooling not supported.")

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
