import torch.optim as optim
from config.config import args
# from MGNN_main.MGNN_Node.main import *
from torch_geometric.data import Data
import os.path as osp
from mgnn_layer import *


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = MotifConv(dataset.num_features, hidden_dim1, compress_dims[0], rel_type, dataset_s, motif_mats,
                               mw_initializer, att_act, motif_dropout, att_dropout, aggr='sum')

        if len(compress_dims) > 1:
            self.conv2 = MotifConv(13 * compress_dims[0], hidden_dim2, compress_dims[1], rel_type, dataset_s,
                                   motif_mats,
                                   mw_initializer, att_act, motif_dropout, att_dropout, aggr='sum')
        else:
            self.register_parameter('conv2', None)

        self.dense = Linear(13 * compress_dims[-1], dataset.num_classes,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer)

    def forward(self, g, h):
        h = F.dropout(h, p=layer_dropout[0], training=self.training)

        h = self.conv1(g, h)
        h = F.relu(h)

        h = F.dropout(h, p=layer_dropout[1], training=self.training)

        if self.conv2 is not None:
            h = self.conv2(g, h)
            h = F.relu(h)

        h = self.dense(h)

        return F.log_softmax(h, dim=1)
