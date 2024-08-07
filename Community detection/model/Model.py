from dhg.models import HGNNP
from sklearn.cluster import kmeans_plusplus
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch
from dhg.nn import HGNNPConv


class HGNNP_GCN(nn.Module):
    def __init__(self, in_ch, n_hid, out_ch, dropout):
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, n_hid)
        self.gcn2 = GCNConv(n_hid, out_ch)
        self.hgnnp1 = HGNNPConv(in_ch, n_hid)
        self.hgnnp2 = HGNNPConv(n_hid, out_ch, is_last=True)  # 不使用非线性函数和dropout
        # self.mlp = nn.Linear(out_ch * 2, out_ch)

    def forward(self, x, edge_index, hg):
        x1 = F.relu(self.gcn1(x, edge_index))
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gcn2(x1, edge_index)
        x3 = self.hgnnp1(x, hg)
        x4 = self.hgnnp2(x3, hg)
        # x = torch.cat((x2, x4), 1)
        # x = self.mlp(x)
        x = x2 / 2 + x4 / 2
        return x


class GCN(nn.Module):
    def __init__(self, in_ch, n_hid, out_ch, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, n_hid)
        self.gcn2 = GCNConv(n_hid, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))  # self.gcn1(x, edge_index).relu()
        x = F.dropout(x, self.dropout)
        x = self.gcn2(x, edge_index)
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


class GCN_motif(nn.Module):
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


def cluster(data, k, num_iter, init=None, cluster_temp=5):
    """
    pytorch (differentiable) implementation of soft k-means clustering.
    """
    # normalize x so it lies on the unit sphere
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
    embeds = torch.diag(1. / torch.norm(embeds, p=2, dim=1)) @ embeds
    data_np = embeds.detach().numpy()
    # data_np[np.isnan(data_np)] = 0
    """data_np[init_indices], (k, fea), (k,), 第二个参数是节点索引，可以根据索引提取张量"""
    mu_init, init_indices = kmeans_plusplus(data_np, k, random_state=None)
    mu_init = torch.tensor(mu_init, requires_grad=True)
    return mu_init


class GCNClusterNet(nn.Module):
    # def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):
    def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout):
        super(GCNClusterNet, self).__init__()
        self.GCN = GCN(nfeat, nhid, nout, dropout)
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


class HGNNPClusterNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):
        super(HGNNPClusterNet, self).__init__()
        self.HGNNP = HGNNP(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init = init
        # torch.manual_seed(123)
        # self.init = torch.rand(self.K, nout)

    def forward(self, x, hg, num_iter=1):
        embeds = self.HGNNP(x, hg)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


class HGNNP_GCNClusterNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout):
        super(HGNNP_GCNClusterNet, self).__init__()
        self.HGNNP_GCN = HGNNP_GCN(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        # self.init = init
        # torch.manual_seed(123)
        # self.init = torch.rand(self.K, nout)

    def forward(self, x, edge_index, hg, num_iter=1):
        embeds = self.HGNNP_GCN(x, edge_index, hg)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


class GCN_motifClusterNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):
        super(GCN_motifClusterNet, self).__init__()
        self.GCN_motif = GCN_motif(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init = init
        # torch.manual_seed(123)
        # self.init = torch.rand(self.K, nout)

    def forward(self, x, edge_index, weight, num_iter=1):
        embeds = self.GCN_motif(x, edge_index, weight)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist
