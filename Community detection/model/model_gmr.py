from sklearn.cluster import kmeans_plusplus
from torch import nn
from torch_geometric.nn.models import GCN
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from torch_geometric.nn import Linear, GCNConv, HypergraphConv


class GCN1(nn.Module):
    def __init__(self, in_ch, n_hid, out_ch, dropout):
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
        x = self.initial(x)
        x = self.convs(x=x, edge_index=edge_index)
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
        self.lin = nn.Linear(nhid, data.num_classes)  # #####################输出层维度要改
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
        return x  # F.log_softmax(x, dim=-1)

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

        return x  # F.log_softmax(x, dim=1)


class GCNClusterNet(nn.Module):
    # def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):  # 随机初始化聚类中心才有init
    def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout):
        super(GCNClusterNet, self).__init__()
        self.GCN = GCN1(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        # self.init = init
        # torch.manual_seed(123)
        # self.init = torch.rand(self.K, nout)

    def forward(self, x, edge_index, num_iter=1):
        embeds = self.GCN(x, edge_index)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


class RAHGClusterNet(nn.Module):
    # def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):  # 随机初始化聚类中心才有init
    def __init__(self, data, nhid, gnn_type, heads, dropout, K, cluster_temp, nout):
        super(RAHGClusterNet, self).__init__()
        self.RAHG = AttentionModel(data, nhid, gnn_type, heads, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp

    def forward(self, data, args, num_iter=1):
        embeds = self.RAHG(data, args)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


def cluster(data, k, num_iter, init=None, cluster_temp=5):
    data = torch.diag(1. / torch.norm(data, p=2, dim=1)) @ data
    # use kmeans++ initialization if nothing is provided
    if init is None:
        # data_np = data.detach().numpy()
        # norm = (data_np ** 2).sum(axis=1)
        # init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0:
            return init
    mu = init  # 生成几个聚类中心（聚类中心维度是输出节点维度，但是其他节点经过拼接）
    for t in range(num_iter):
        # get distances between all data points and cluster centers
        dist = data @ mu.t()
        # cluster responsibilities via softmax
        r = torch.softmax(cluster_temp * dist, 1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)  # cluster_temp控制平滑程度， 平衡集群的紧密程度
    return mu, r, dist


def cluster_(embeds, k):
    embeds = torch.diag(1. / torch.norm(embeds, p=2, dim=1)) @ embeds  # 当时加数据归一化，就是因为超出精度
    data_np = embeds.detach().numpy()
    # data_np[np.isnan(data_np)] = 0
    """data_np[init_indices], (k, fea), (k,), 第二个参数是节点索引，可以根据索引提取张量"""
    mu_init, init_indices = kmeans_plusplus(data_np, k, random_state=None)
    mu_init = torch.tensor(mu_init, requires_grad=True)
    # device = 'cuda:0'  # 超算这里的操作是不同的，设备投影
    # mu_init = torch.tensor(mu_init, requires_grad=True).to(device)
    return mu_init
