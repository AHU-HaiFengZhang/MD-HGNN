from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch
from dhg.nn import HGNNPConv
import torch
from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.nn import MessagePassing, Linear, GCNConv, HypergraphConv, GCN2Conv, PointNetConv
from torch_geometric.data import Data


class HGNNP_GCN(nn.Module):
    def __init__(self, in_ch, n_hid, out_ch, dropout):
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, n_hid)
        self.gcn2 = GCNConv(n_hid, out_ch)
        self.hgnnp1 = HGNNPConv(in_ch, n_hid)  # 默认dropout 0.5
        self.hgnnp2 = HGNNPConv(n_hid, out_ch, is_last=True)  # 不使用非线性函数和dropout
        self.mlp = nn.Linear(out_ch * 2, out_ch)

    def forward(self, x, edge_index, hg):
        x1 = F.relu(self.gcn1(x, edge_index))
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gcn2(x1, edge_index)
        x3 = self.hgnnp1(x, hg)
        x4 = self.hgnnp2(x3, hg)
        # x = torch.cat((x2, x4), 1)  # 沿指定维度进行拼接
        # x = self.mlp(x)
        x = x2 / 2 + x4 / 2
        return x


class GCN1(nn.Module):
    def __init__(self, in_ch, n_hid, out_ch, dropout):  # 参数顺序应用
        super(GCN1, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, n_hid)
        self.gcn2 = GCNConv(n_hid, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))  # self.gcn1(x, edge_index).relu()
        x = F.dropout(x, self.dropout)
        x = self.gcn2(x, edge_index)
        return x


class MyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers, dropout):
        super(MyGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.initial = Linear(in_channels=self.in_channels, out_channels=self.hidden_channels, bias=True,
                              weight_initializer='glorot')
        self.convs = GCN(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels,
                         out_channels=self.hidden_channels, num_layers=self.num_layers, dropout=self.dropout)
        self.final = Linear(in_channels=self.hidden_channels, out_channels=self.out_channels, bias=True,
                            weight_initializer='glorot')

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.initial(x)
        x = self.convs(x=x, edge_index=edge_index)
        # Pooling.
        # x = global_add_pool(x, batch=batch).squeeze()
        # Classification.
        x = self.final(x).squeeze()
        return x

#
#
#
#


class AttentionModel(nn.Module):  # RAHG model
    def __init__(self, data, nhid, gnn_type, heads, dropout):
        super().__init__()
        r_feats_num = data.num_node_features_r
        s_feats_num = data.num_node_features_s
        self.heads = heads
        self.linear_r = nn.Linear(r_feats_num, nhid)
        self.q_r = nn.Parameter(torch.rand(nhid, self.heads))
        self.linear_s = nn.Linear(s_feats_num, nhid)
        self.q_s = nn.Parameter(torch.rand(nhid, self.heads))
        self.att_fusion = nn.Linear(self.heads * nhid, nhid)

        self.sim_fusion = nn.Linear(r_feats_num + s_feats_num, nhid)
        if gnn_type == 'gcn':
            self.model = GCN_R(data, nhid)
        if gnn_type == 'hgcn':
            self.model = HyperGCN(data, nhid, dropout=dropout)

    def forward(self, data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha
            fusion_x = torch.cat(
                [r_alpha[:, i].view(-1, 1) * r_feats + s_alpha[:, i].view(-1, 1) * s_feats for i in range(self.heads)],
                dim=1)
            fusion_x = self.att_fusion(fusion_x)

        if args.fusion_type == 'concat':
            fusion_x = torch.cat((r_feats, s_feats), dim=1)
            fusion_x = self.sim_fusion(fusion_x)
        data.x = fusion_x
        result = self.model.forward(data, args)
        return result

    def predict(self, data: Data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha
            fusion_x = torch.cat(
                [r_alpha[:, i].view(-1, 1) * r_feats + s_alpha[:, i].view(-1, 1) * s_feats for i in range(self.heads)],
                dim=1)
            fusion_x = self.att_fusion(fusion_x)

        if args.fusion_type == 'concat':
            fusion_x = torch.cat((r_feats, s_feats), dim=1)
            fusion_x = self.sim_fusion(fusion_x)
        data.x = fusion_x
        hid = self.model.predict(data)
        return hid

    def predict_att_weights(self, data: Data, args):
        r_feats, s_feats = data.x_role, data.x_stru
        if args.fusion_type == 'attention':
            r_feats = torch.tanh(self.linear_r(r_feats))
            r_alpha = torch.matmul(r_feats, self.q_r)

            s_feats = torch.tanh(self.linear_s(s_feats))
            s_alpha = torch.matmul(s_feats, self.q_s)

            alpha = torch.exp(r_alpha) + torch.exp(s_alpha)
            r_alpha = torch.exp(r_alpha) / alpha
            s_alpha = torch.exp(s_alpha) / alpha

            return r_alpha, s_alpha
        else:
            return

    def reset_parameters(self):
        self.linear_r.reset_parameters()
        self.linear_s.reset_parameters()


class HyperGCN(nn.Module):
    def __init__(self, data, nhid, dropout=0.5):
        super(HyperGCN, self).__init__()
        self.dropout = dropout
        # self.HGC1 = HypergraphConv(data.num_node_features+nhid, nhid)
        self.HGC1 = HypergraphConv(nhid, nhid)
        self.HGC2 = HypergraphConv(nhid, nhid)
        self.lin = nn.Linear(nhid, data.num_classes)
        # self.down_sample = nn.Linear(data.num_node_features+nhid, nhid)
        self.down_sample = nn.Linear(nhid, nhid)

    def forward(self, data: Data, args):
        x, H = data.x, data.H
        # x=torch.cat((data.x,data.fx),dim=1)
        origin_x = self.down_sample(x)
        x = self.HGC1(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.HGC2(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if args.residual:
            x = x + origin_x
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    def predict(self, data: Data):
        x, H = data.x, data.H
        x = self.HGC1(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.HGC2(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_R(nn.Module):
    def __init__(self, data, nhid):
        super().__init__()
        self.gcn1 = GCNConv(data.num_node_features, nhid)
        self.gcn2 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, data.num_classes)
        self.down_sample = nn.Linear(data.num_node_features, nhid)

    def forward(self, data: Data):
        x, adj = data.x, data.edge_index
        x = self.gcn1(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, adj)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        # x = x + origin_x
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


class GCN_motif(nn.Module):  # GCN_Am, GCN_AM model
    def __init__(self, in_ch, n_hid, out_ch, dropout):
        super(GCN_motif, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, n_hid)
        self.gcn2 = GCNConv(n_hid, out_ch)

    def forward(self, x, edge_index, weight):
        x = F.relu(self.gcn1(x, edge_index, weight))
        x = F.dropout(x, self.dropout)
        x = self.gcn2(x, edge_index, weight)
        return x
