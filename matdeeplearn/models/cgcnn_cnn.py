import torch
import wandb
from torch.nn import AdaptiveAvgPool1d, BatchNorm1d, Dropout, Linear, ReLU, Sequential

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
        # save model file
        wandb.run.log_code(".", include_fn=lambda path: path.endswith("cgcnn_cnn.py"))

        # set up CNN
        self.conv_layer1 = Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
            ReLU(),
            BatchNorm1d(8),
            AdaptiveAvgPool1d(100),
        )
        self.conv_layer2 = Sequential(
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(50),
        )
        self.conv_layer3 = Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(25),
        )
        self.conv_layer4 = Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(12),
        )
        self.conv_layer5 = Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(6),
        )
        self.conv_layer6 = Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(1),
            Dropout(0.00),
        )
        self.ff_layer1 = Sequential(Linear(128, 128), ReLU())
        self.ff_layer2 = Sequential(Linear(128, 128), ReLU())

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

    def forward(self, data):
        # get DOS out
        dos_unscaled = data.scaled * data.scaling_factor.view(-1, 1).expand_as(
            data.scaled
        )
        dos_out = self.conv_layer1(dos_unscaled.unsqueeze(1))
        dos_out = self.conv_layer2(dos_out)
        dos_out = self.conv_layer3(dos_out)
        dos_out = self.conv_layer4(dos_out)
        dos_out = self.conv_layer5(dos_out)
        dos_out = self.conv_layer6(dos_out)
        dos_out = self.ff_layer1(dos_out.squeeze(2))
        dos_out = self.ff_layer2(dos_out)
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

        out = self._forward_gnn_layers(data, out)
        out = self._forward_post_gnn_layers(data, out)

        out = out.view(-1) if out.shape[1] == 1 else out
        return out
