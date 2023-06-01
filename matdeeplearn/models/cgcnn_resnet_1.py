import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import AdaptiveAvgPool1d, BatchNorm1d, Dropout, Linear, ReLU, Sequential

import wandb
from matdeeplearn.common.registry import registry
from matdeeplearn.models.cgcnn import CGCNN


@registry.register_model("CGCNN_RESNET_1")
class CGCNN_RESNET_1(CGCNN):
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
        super(CGCNN_RESNET_1, self).__init__(
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
        # self.setup_cnn()
        # self.cnn = ResNet1D(
        #     in_channels=3,
        #     base_filters=64,
        #     kernel_size=3,
        #     stride=2,
        #     groups=1,
        #     n_block=4,
        #     n_classes=128,
        # )

        self.cnn = ResNet(BasicBlock, [3, 4, 6, 3])

    def setup_cnn(self):
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
        # dos_out = self.cnn_forward(dos_unscaled)
        dos_out = self.cnn(dos_unscaled)

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

    def cnn_forward(self, dos_unscaled):
        dos_out = self.conv_layer1(dos_unscaled)
        dos_out = self.conv_layer2(dos_out)
        dos_out = self.conv_layer3(dos_out)
        dos_out = self.conv_layer4(dos_out)
        dos_out = self.conv_layer5(dos_out)
        dos_out = self.conv_layer6(dos_out)
        dos_out = self.ff_layer1(dos_out.squeeze(2))
        dos_out = self.ff_layer2(dos_out)
        return dos_out


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # self.conv_layer1 = Sequential(
        #     torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5),
        #     ReLU(),
        #     BatchNorm1d(16),
        #     AdaptiveAvgPool1d(100),
        # )

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        downsample,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1

        # the first conv
        self.conv1 = nn.Sequential(
            MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                groups=1,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            MyConv1dPadSame(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                groups=self.groups,
            ),
            nn.BatchNorm1d(out_channels),
        )

        self.relu = nn.ReLU()

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.downsample(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=128):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(channels),
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.inplanes = channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x.squeeze(2))

        return x


# class ResNet1D(nn.Module):
#     """
#
#     Input:
#         X: (n_samples, n_channel, n_length)
#         Y: (n_samples)
#
#     Output:
#         out: (n_samples)
#
#     Pararmetes:
#         in_channels: dim of input, the same as n_channel
#         base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
#         kernel_size: width of kernel
#         stride: stride of kernel moving
#         groups: set larget to 1 as ResNeXt
#         n_block: number of blocks
#         n_classes: number of classes
#
#     """
#
#     def __init__(
#         self,
#         in_channels,
#         base_filters,
#         kernel_size,
#         stride,
#         groups,
#         n_block,
#         n_classes,
#         downsample_gap=2,
#         increasefilter_gap=4,
#         use_bn=True,
#         use_do=True,
#         verbose=False,
#     ):
#         super(ResNet1D, self).__init__()
#
#         self.verbose = verbose
#         self.n_block = n_block
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.use_bn = use_bn
#         self.use_do = use_do
#
#         self.downsample_gap = downsample_gap  # 2 for base model
#         self.increasefilter_gap = increasefilter_gap  # 4 for base model
#
#         # first block
#         self.first_block = nn.Sequential(
#             MyConv1dPadSame(
#                 in_channels=in_channels,
#                 out_channels=base_filters,
#                 kernel_size=self.kernel_size,
#                 stride=1,
#             ),
#             nn.BatchNorm1d(base_filters),
#             nn.ReLU(),
#         )
#
#         out_channels = base_filters
#
#         # residual blocks
#         self.basicblock_list = nn.ModuleList()
#         for i_block in range(self.n_block):
#             # is_first_block
#             if i_block == 0:
#                 is_first_block = True
#             else:
#                 is_first_block = False
#             # downsample at every self.downsample_gap blocks
#             if i_block % self.downsample_gap == 1:
#                 downsample = True
#             else:
#                 downsample = False
#             # in_channels and out_channels
#             if is_first_block:
#                 in_channels = base_filters
#                 out_channels = in_channels
#             else:
#                 # increase filters at every self.increasefilter_gap blocks
#                 in_channels = int(
#                     base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
#                 )
#                 if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
#                     out_channels = in_channels * 2
#                 else:
#                     out_channels = in_channels
#
#             tmp_block = BasicBlock(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 groups=self.groups,
#                 downsample=downsample,
#                 use_bn=self.use_bn,
#                 use_do=self.use_do,
#                 is_first_block=is_first_block,
#             )
#             self.basicblock_list.append(tmp_block)
#
#         # final prediction
#         self.final_bn = nn.BatchNorm1d(out_channels)
#         self.final_relu = nn.ReLU(inplace=True)
#         # self.do = nn.Dropout(p=0.5)
#         self.dense = nn.Linear(out_channels, n_classes)
#         # self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#
#         out = x
#
#         # first conv
#         out = self.first_block(out)
#
#         # residual blocks, every block has two conv
#         for i_block in range(self.n_block):
#             net = self.basicblock_list[i_block]
#             if self.verbose:
#                 print(
#                     "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
#                         i_block, net.in_channels, net.out_channels, net.downsample
#                     )
#                 )
#             out = net(out)
#             if self.verbose:
#                 print(out.shape)
#
#         # final prediction
#         if self.use_bn:
#             out = self.final_bn(out)
#         out = self.final_relu(out)
#         out = out.mean(-1)
#         if self.verbose:
#             print("final pooling", out.shape)
#         # out = self.do(out)
#         out = self.dense(out)
#         if self.verbose:
#             print("dense", out.shape)
#         # out = self.softmax(out)
#         if self.verbose:
#             print("softmax", out.shape)
#
#         return out
