import torch
from torch.nn import (
    AdaptiveAvgPool1d,
    AdaptiveMaxPool1d,
    BatchNorm1d,
    Dropout,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
)

import wandb
from matdeeplearn.common.registry import registry
from matdeeplearn.models.cgcnn import CGCNN


@registry.register_model("CGCNN_CNN")
class CGCNN_CNN(CGCNN):
    def __init__(
        self,
        edge_steps,
        self_loop,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(CGCNN_CNN, self).__init__(
            edge_steps,
            self_loop,
            data,
            dim1,
            dim2,
            pre_fc_count,
            gc_count,
            post_fc_count,
            pool,
            pool_order,
            batch_norm,
            batch_track_stats,
            act,
            dropout_rate,
            **kwargs
        )

        # set up CNN
        self.conv_layer1 = Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5),
            ReLU(),
            BatchNorm1d(16),
            AdaptiveAvgPool1d(100),
        )
        self.conv_layer2 = Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            ReLU(),
            BatchNorm1d(32),
            AdaptiveAvgPool1d(50),
        )
        self.conv_layer3 = Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            BatchNorm1d(64),
            AdaptiveAvgPool1d(25),
        )
        self.conv_layer4 = Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            ReLU(),
            BatchNorm1d(128),
            AdaptiveAvgPool1d(12),
        )
        self.conv_layer5 = Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU(),
            BatchNorm1d(256),
            AdaptiveAvgPool1d(6),
        )
        self.conv_layer6 = Sequential(
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            ReLU(),
            BatchNorm1d(512),
            AdaptiveAvgPool1d(1),
            Dropout(0.05),
        )
        self.ff_layer1 = Sequential(Linear(512, 128), BatchNorm1d(128), ReLU())
        self.ff_layer2 = Sequential(Linear(128, 128), BatchNorm1d(128), ReLU())

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer).
        Note the size for lin is +128 to account for the DOS"""
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = Linear(self.num_features + 128, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = Linear(self.post_fc_dim * 2, self.dim2)
                    else:
                        lin = Linear(self.post_fc_dim + 128, self.dim2)
                else:
                    lin = Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out

    def forward(self, data):
        # get DOS out
        # dos_unscaled = torch.swapaxes(data.scaled * data.scaling_factor.view(-1, 1, 1).expand_as(data.scaled), 1,2)
        dos_unscaled = data.scaled * data.scaling_factor.view(-1, 1, 1).expand_as(
            data.scaled
        )
        dos_out = self.conv_layer1(dos_unscaled)
        dos_out = self.conv_layer2(dos_out)
        dos_out = self.conv_layer3(dos_out)
        dos_out = self.conv_layer4(dos_out)
        dos_out = self.conv_layer5(dos_out)
        dos_out = self.conv_layer6(dos_out)
        dos_out = self.ff_layer1(dos_out.squeeze(2))
        dos_out = self.ff_layer2(dos_out)
        # print(dos_out[0])
        # print(data.scaled.unsqueeze(1).shape, dos_out.shape)

        # Pre-GNN dense layers
        if len(self.pre_lin_list) == 0:
            out = data.x
            # if data.x can be a float, then don't need this if/else statement
        else:
            # add DOS to data
            # SHOULD DOS ALWAYS BE ADDED - EVEN IF NO PRE_LIN_LIST??
            out = self._forward_pre_gnn_layers(
                torch.cat((data.x.float(), dos_out.squeeze(-1)), 1)
            )
            # out = self._forward_pre_gnn_layers(data.x.float())

        out = self._forward_gnn_layers(data, out)
        # out = self._forward_post_gnn_layers(data, out)
        out = self._forward_post_gnn_layers(
            data, torch.cat((out.float(), dos_out.squeeze(-1)), 1)
        )

        out = out.view(-1) if out.shape[1] == 1 else out
        return out
