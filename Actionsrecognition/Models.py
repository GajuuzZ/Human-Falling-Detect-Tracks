### Reference from: https://github.com/yysijie/st-gcn/tree/master/net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Actionsrecognition.Utils import Graph


class GraphConvolution(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        - in_channel: (int) Number of channels in the input sequence data.
        - out_channels: (int) Number of channels produced by the convolution.
        - kernel_size: (int) Size of the graph convolving kernel.
        - t_kernel_size: (int) Size of the temporal convolving kernel.
        - t_stride: (int, optional) Stride of the temporal convolution. Default: 1
        - t_padding: (int, optional) Temporal zero-padding added to both sides of
            the input. Default: 0
        - t_dilation: (int, optional) Spacing between temporal kernel elements. Default: 1
        - bias: (bool, optional) If `True`, adds a learnable bias to the output.
            Default: `True`
    Shape:
        - Inputs x: Graph sequence in :math:`(N, in_channels, T_{in}, V)`,
                 A: Graph adjacency matrix in :math:`(K, V, V)`,
        - Output: Graph sequence out in :math:`(N, out_channels, T_{out}, V)`

            where
                :math:`N` is a batch size,
                :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
                :math:`T_{in}/T_{out}` is a length of input/output sequence,
                :math:`V` is the number of graph nodes.

    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        - in_channels: (int) Number of channels in the input sequence data.
        - out_channels: (int) Number of channels produced by the convolution.
        - kernel_size: (tuple) Size of the temporal convolving kernel and
            graph convolving kernel.
        - stride: (int, optional) Stride of the temporal convolution. Default: 1
        - dropout: (int, optional) Dropout rate of the final output. Default: 0
        - residual: (bool, optional) If `True`, applies a residual mechanism.
            Default: `True`
    Shape:
        - Inputs x: Graph sequence in :math: `(N, in_channels, T_{in}, V)`,
                 A: Graph Adjecency matrix in :math: `(K, V, V)`,
        - Output: Graph sequence out in :math: `(N, out_channels, T_{out}, V)`
            where
                :math:`N` is a batch size,
                :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
                :math:`T_{in}/T_{out}` is a length of input/output sequence,
                :math:`V` is the number of graph nodes.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True)
                                 )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels)
                                          )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


class StreamSpatialTemporalGraph(nn.Module):
    """Spatial temporal graph convolutional networks.
    Args:
        - in_channels: (int) Number of input channels.
        - graph_args: (dict) Args map of `Actionsrecognition.Utils.Graph` Class.
        - num_class: (int) Number of class outputs. If `None` return pooling features of
            the last st-gcn layer instead.
        - edge_importance_weighting: (bool) If `True`, adds a learnable importance
            weighting to the edges of the graph.
        - **kwargs: (optional) Other parameters for graph convolution units.
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
        or If num_class is `None`: `(N, out_channels)`
            :math:`out_channels` is number of out_channels of the last layer.
    """
    def __init__(self, in_channels, graph_args, num_class=None,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        # Load graph.
        graph = Graph(**graph_args)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Networks.
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))

        # initialize parameters for edge importance weighting.
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        if num_class is not None:
            self.cls = nn.Conv2d(256, num_class, kernel_size=1)
        else:
            self.cls = lambda x: x

    def forward(self, x):
        # data normalization.
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forward.
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = self.cls(x)
        x = x.view(x.size(0), -1)

        return x


class TwoStreamSpatialTemporalGraph(nn.Module):
    """Two inputs spatial temporal graph convolutional networks.
    Args:
        - graph_args: (dict) Args map of `Actionsrecognition.Utils.Graph` Class.
        - num_class: (int) Number of class outputs.
        - edge_importance_weighting: (bool) If `True`, adds a learnable importance
            weighting to the edges of the graph.
        - **kwargs: (optional) Other parameters for graph convolution units.
    Shape:
        - Input: :tuple of math:`((N, 3, T, V), (N, 2, T, V))`
        for points and motions stream where.
            :math:`N` is a batch size,
            :math:`in_channels` is data channels (3 is (x, y, score)), (2 is (mot_x, mot_y))
            :math:`T` is a length of input sequence,
            :math:`V` is the number of graph nodes,
        - Output: :math:`(N, num_class)`
    """
    def __init__(self, graph_args, num_class, edge_importance_weighting=True,
                 **kwargs):
        super().__init__()
        self.pts_stream = StreamSpatialTemporalGraph(3, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)
        self.mot_stream = StreamSpatialTemporalGraph(2, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)

        self.fcn = nn.Linear(256 * 2, num_class)

    def forward(self, inputs):
        out1 = self.pts_stream(inputs[0])
        out2 = self.mot_stream(inputs[1])

        concat = torch.cat([out1, out2], dim=-1)
        out = self.fcn(concat)

        return torch.sigmoid(out)
