import math

import torch
import torch.nn.functional as F
from torch import nn


class RTLayer(nn.Module):
    def __init__(
        self,
        node_dim: int = None,
        node_hidden: int = None,
        edge_dim: int = None,
        edge_hidden_1: int = None,
        edge_hidden_2: int = None,
        heads: int = None,
        dropout: float = None,
        disable_edge_updates: bool = False,
    ) -> None:
        """Relational transformer attention layer

        Args:
            node_dim (int, optional): node feature dimension. Defaults to None.
            node_hidden (int, optional): hidden dimension for node features. Defaults to None.
            edge_dim (int, optional): edge feature dimension. Defaults to None.
            edge_hidden_1 (int, optional): first edge hidden dim. Defaults to None.
            edge_hidden_2 (int, optional): second edge hidden dim. Defaults to None.
            heads (int, optional): number of attention heads for MHA. Defaults to None.
            dropout (float, optional): dropout probability. Defaults to None.
            disable_edge_updates (bool, optional): whether or not to update edge features (node features always updated). Defaults to False.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_hidden = node_hidden
        self.edge_dim = edge_dim
        self.edge_hidden_1 = edge_hidden_1
        self.edge_hidden_2 = edge_hidden_2
        self.heads = heads
        assert (
            node_hidden % heads == 0
        ), "Node hidden dimension must be divisible by number of heads"
        self.multihead_hidden = node_hidden // heads
        self.dropout = dropout
        # for ablation
        self.disable_edge_updates = disable_edge_updates
        # node layers
        self.attn_layer = RTAttention(
            self.node_dim, self.heads, self.multihead_hidden, self.edge_dim
        )
        self.node_fc_1 = nn.Linear(self.node_dim, self.node_dim)
        self.node_ln_1 = nn.LayerNorm(self.node_dim)
        self.node_drop_1 = nn.Dropout(self.dropout)
        self.node_fc_2 = nn.Linear(self.node_dim, self.node_hidden)
        self.node_fc_3 = nn.Linear(self.node_hidden, self.node_dim)
        self.node_drop_2 = nn.Dropout(self.dropout)
        self.node_ln_2 = nn.LayerNorm(self.node_dim)
        # edge layers
        self.edge_fc_1 = nn.Linear(self.edge_hidden_1, self.edge_dim)
        self.edge_fc_2 = nn.Linear(self.edge_dim, self.edge_dim)
        self.edge_drop_1 = nn.Dropout(self.dropout)
        self.edge_ln_1 = nn.LayerNorm(self.edge_dim)
        self.edge_fc_3 = nn.Linear(self.edge_dim, self.edge_hidden_2)
        self.edge_fc_4 = nn.Linear(self.edge_hidden_2, self.edge_dim)
        self.edge_drop_2 = nn.Dropout(self.dropout)
        self.edge_ln_2 = nn.LayerNorm(self.edge_dim)

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor) -> None:
        # node updates
        attn_node_outs = self.attn_layer(x, dense_adj)
        residuals = self.node_fc_1(attn_node_outs)
        residuals = self.node_drop_1(residuals)
        x = self.node_ln_1(x + residuals)
        # second residual set
        residuals = self.node_fc_2(x)
        residuals = self.node_fc_3(F.relu(residuals))
        residuals = self.node_drop_2(residuals)
        x = self.node_drop_2(x + residuals)
        x = self.node_ln_2(x)

        # edge updates
        if not self.disable_edge_updates:
            # (B, N, N, Nd)
            source_nodes = x.unsqueeze(1)
            expanded_source_nodes = torch.tile(source_nodes, (1, x.size(1), 1, 1))
            target_nodes = x.unsqueeze(2)
            expanded_target_nodes = torch.tile(target_nodes, (1, 1, x.size(1), 1))
            # (B, N, N, Ed)
            reversed_edges = torch.swapaxes(dense_adj, -2, -3)
            # edge update args
            update_args = torch.cat(
                [
                    dense_adj,
                    reversed_edges,
                    expanded_source_nodes,
                    expanded_target_nodes,
                ],
                dim=-1,
            )
            # compute edge updates
            residuals = self.edge_fc_1(update_args)
            residuals = self.edge_fc_2(F.relu(residuals))
            residuals = self.edge_drop_1(residuals)
            dense_adj = self.edge_ln_1(dense_adj + residuals)
            # second residual set
            residuals = self.edge_fc_3(dense_adj)
            residuals = self.edge_fc_4(F.relu(residuals))
            residuals = self.edge_drop_2(residuals)
            dense_adj = self.edge_ln_2(dense_adj + residuals)

        return x, dense_adj


class RTAttention(nn.Module):
    def __init__(self, node_dim, heads, hidden_dim, edge_dim) -> None:
        """RT attention module

        Args:
            node_dim (int): node hidden dimension
            heads (int): number of attention heads
            hidden_dim (int): hidden dim to use in FFN
            edge_dim (int): edge feature dimension
        """
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)
        # query, key, value projections for nodes
        self.wnq = nn.Linear(node_dim, node_dim, bias=False)
        self.wnk = nn.Linear(node_dim, node_dim, bias=False)
        self.wnv = nn.Linear(node_dim, node_dim, bias=False)
        # query, key, value projections for edges
        self.weq = nn.Linear(edge_dim, node_dim, bias=False)
        self.wek = nn.Linear(edge_dim, node_dim, bias=False)
        self.wev = nn.Linear(edge_dim, node_dim, bias=False)

    def reshape_nodes_to_batches(self, x: torch.Tensor, query=False) -> torch.Tensor:
        """
        Adapted from https://github.com/CyberZHG/torch-multi-head-attention
        """
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim % self.heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        sub_dim = embed_dim // self.heads
        if query:
            return (
                x.reshape(batch_size, seq_len, self.heads, sub_dim)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * self.heads, seq_len, 1, sub_dim)
            )
        else:
            return (
                x.reshape(batch_size, seq_len, self.heads, sub_dim)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * self.heads, 1, seq_len, sub_dim)
            )

    def reshape_edges_to_batches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapted from https://github.com/CyberZHG/torch-multi-head-attention
        """
        batch_size, seq_len, seq_len, embed_dim = x.size()
        sub_dim = embed_dim // self.heads
        return (
            x.reshape(batch_size, seq_len, seq_len, self.heads, sub_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch_size * self.heads, seq_len, seq_len, sub_dim)
        )

    def reshape_from_batches(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, sub_dim = x.size()
        batch_size //= self.heads
        return (
            x.reshape(batch_size, self.heads, seq_len, sub_dim)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, seq_len, sub_dim * self.heads)
        )

    def forward(
        self,
        x: torch.Tensor,
        dense_adj: torch.Tensor,
    ) -> torch.Tensor:
        # compute (B * H, N, 1, Nd / H) + (B * H, N, N, Nd / H) via broadcasting
        query = self.reshape_nodes_to_batches(
            self.wnq(x), query=True
        ) + self.reshape_edges_to_batches(self.weq(dense_adj))
        key = self.reshape_nodes_to_batches(
            self.wnk(x), query=False
        ) + self.reshape_edges_to_batches(self.wek(dense_adj))
        value = self.reshape_nodes_to_batches(
            self.wnv(x)
        ) + self.reshape_edges_to_batches(self.wev(dense_adj))

        # attention score computation, (B * H, N, N, 1, Nd / H) * (B * H, N, N, Nd / H, 1)
        scores = torch.matmul(query.unsqueeze(-2), key.unsqueeze(-1)) * self.scale
        attn = F.softmax(scores.squeeze(-2), dim=-2)

        # (B * H, N, 1, N, 1) * (B * H, N, N, 1, Nd / H)
        attn = attn.unsqueeze(-3).permute(0, 1, 4, 2, 3)
        value = value.unsqueeze(-2).permute(0, 1, 4, 2, 3)
        new_nodes = torch.matmul(attn, value).squeeze(-1).squeeze(-1)

        # (B, N, Nd)
        return self.reshape_from_batches(new_nodes)
