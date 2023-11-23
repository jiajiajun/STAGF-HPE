import torch
import torch.nn as nn
import numpy as np


inter_channels = [128, 128, 256]

fc_out = inter_channels[-1]
fc_unit = 512

class t_gcn(nn.Module):
    """

    Args:
        in_channels (int): Number of channels in the input data
        cat: True: concatinate coarse and fine features
            False: add coarse and fine features

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes for each frame,
            :math:`M_{in}` is the number of instance in a frame. (In this task always equals to 1)
    Return:
        out_all_frame: True: return all frames 3D results
                        False: return target frame result

        x_out: final output.

    """

    def __init__(self, adj, input_dim, output_dim, pad):
        super().__init__()

        # load graph
        self.momentum = 0.1
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.cat = True
        self.inplace = True

        # original graph
        self.graph = Graph(adj, pad)
        # get adjacency matrix of K clusters
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda() # K, T*V, T*V

        # build networks
        kernel_size = self.A.size(0)
        num_joints = adj.shape[0]

        self.data_bn = nn.BatchNorm1d(self.in_channels * num_joints, self.momentum)
        self.tgcn = tgcn(self.in_channels, self.out_channels, kernel_size)


    def forward(self, x, out_all_frame=False):

        # data normalization
        N, T, V, C = x.size()

        x = x.permute(0, 2, 3, 1).contiguous() # N, V, C, T
        x = x.view(N, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 1, 3, 2).contiguous() # N, C, T, V
        x = x.view(N, C, 1, -1) #(N * M), C, 1, (T*V)

        # forward GCN
        x, _ = self.tgcn(x, self.A) # (N * M), C, 1, (T*V)

        x = x.view(N, -1, T, V) # N, C, T ,V
        x = x.permute(0, 2, 3, 1).contiguous()  # N, T, V, C

        # output
        if out_all_frame:
            x_out = x
        else:
            x_out= x[:, :, self.pad].unsqueeze(2)
        return x_out


class tgcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size :number of the node clusters

        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, 1, T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes of each frame.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):

        super().__init__()
        self.inplace = True

        self.momentum = 0.1
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)

    def forward(self, x, A):

        x, A = self.gcn(x, A)

        return x, A


class Graph():
    """ The Graph to model the skeletons of human body/hand

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration


        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame


        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 adj,
                 pad):

        self.pad = pad
        self.seqlen = 2*self.pad+1
        self.num_node_each = adj.shape[0]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]


        # get distance of each node to center

        self.get_adjacency(adj)


    def __str__(self):
        return self.A


    def get_adjacency(self, adj):

        A = []
        self.num_node = adj.shape[0]
        a_forward = np.zeros((self.num_node, self.num_node))
        a_back = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            for j in range(self.num_node):
                if (j, i) in self.time_link_forward:
                    a_forward[j, i] = adj[j, i]
                elif (j, i) in self.time_link_back:
                    a_back[j, i] = adj[j, i]

            if self.seqlen > 1:
                A.append(a_forward)
                A.append(a_back)

        A = np.stack(A)
        self.A = A

class ConvTemporalGraphical(nn.Module):

    """The basic module for applying a graph convolution.


    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels,1,  T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size`,
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes per frame.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

