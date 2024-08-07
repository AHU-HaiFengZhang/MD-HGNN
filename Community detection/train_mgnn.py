import torch.optim as optim
from torch_geometric.datasets import Planetoid
from model.model_gmr import *
import os.path as osp
import math
from config.config import args
from layer_mgnn import *
import warnings


warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid('data', name=args.dataset)
# Data(x=[2708, 1433], edge_index=[2, 8448], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_labe
adj = adj_generate(dataset.edge_index).numpy()
adj_all, _ = load_data_1(adj, dataset.x)
bin_adj_all = (adj_all.to_dense() > 0).float()  # standard adjacency matrix
test_object = make_modularity_matrix(bin_adj_all)
bin_adj_all = bin_adj_all.to(device)
test_object = test_object.to(device)
nfeat = dataset.x.shape[1]

data = Data(x=dataset.x, edge_index=dataset.edge_index).to(device)

dataset_s = 'Cora'  # ######################################################################
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
x, edge_index = data.x, data.edge_index
x = x.to(device)
num_edge = edge_index.shape[-1]
num_filter = 1
hidden_dim1 = 128  # 16
compress_dims = [6]  # 8
att_act = torch.sigmoid
layer_dropout = [0.5, 0.6]
motif_dropout = 0.1
att_dropout = 0.1
mw_initializer = 'Xavier_Uniform'
kernel_initializer = None
bias_initializer = None
hidden_dim2 = 64
row, col = edge_index.tolist()
sp_mat = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))  # 不带自环
edge_weight_norm = normalize_adj(sp_mat).data.reshape([-1, 1])

mc = MotifCounter(dataset_s, [sp_mat], osp.join(path, dataset_s, 'processed'))
motif_mats = mc.split_13motif_adjs()
motif_mats = [convert_sparse_matrix_to_th_sparse_tensor(normalize_adj(motif_mat)).to(device) for motif_mat in
              motif_mats]

weight_index_data = np.array([range(num_filter)], dtype=np.int32).repeat(num_edge, axis=0)

rel_type = [str(rel) for rel in set(weight_index_data.flatten().tolist())]
graph_data = {('P', rel, 'P'): [[], []] for rel in rel_type}
edge_data = {rel: [] for rel in rel_type}

for rel in rel_type:
    for eid in range(weight_index_data.shape[0]):
        for j in range(num_filter):
            if str(weight_index_data[eid, j]) == rel:
                graph_data[('P', rel, 'P')][0].append(row[eid])
                graph_data[('P', rel, 'P')][1].append(col[eid])
                edge_data[rel].append([edge_weight_norm[eid, 0]])

graph_data = {rel: tuple(graph_data[rel]) for rel in graph_data}

g = dgl.heterograph(graph_data).int().to(device)
for rel in rel_type:
    g.edges[rel].data['edge_weight_norm'] = torch.tensor(edge_data[rel], dtype=torch.float32).to(device)


class Linear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 kernel_initializer=None,
                 bias_initializer='Zero'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        if kernel_initializer is not None:
            self.reset_parameters(self.linear.weight, kernel_initializer)
        if bias and bias_initializer is not None:
            self.reset_parameters(self.linear.bias, bias_initializer)

    @staticmethod
    def reset_parameters(param, initializer):
        if initializer == 'Xavier_Uniform':
            init.xavier_uniform_(param, gain=1.)
        elif initializer == 'Xavier_Normal':
            init.xavier_normal_(param, gain=1.)
        elif initializer == 'Kaiming_Uniform':
            init.kaiming_uniform_(param)
        elif initializer == 'Kaiming_Normal':
            init.kaiming_normal_(param, a=1.)
        elif initializer == 'Uniform':
            init.uniform_(param, a=0, b=1)
        elif initializer == 'Normal':
            init.normal_(param, mean=0, std=1)
        elif initializer == 'Orthogonal':
            init.orthogonal_(param, gain=1)
        elif initializer == 'Zero':
            init.zeros_(param)
        elif initializer == 'gcn':
            stdv = 1. / math.sqrt(param.size(1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.linear(x)  # # 另一个

# print(nfeat)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # print(dataset.x.shape[1], hidden_dim1)
        self.conv1 = MotifConv(500, hidden_dim1, compress_dims[0], rel_type, dataset_s, motif_mats,
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
        return h # F.log_softmax(h, dim=1)


class MGNNClusterNet(nn.Module):
    # def __init__(self, nfeat, nhid, nout, K, cluster_temp, dropout, init):  # 随机初始化聚类中心才有init
    def __init__(self, K, cluster_temp, nout):
        super(MGNNClusterNet, self).__init__()
        self.MGNN = Net()
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp

    def forward(self, g, x, num_iter=1):
        embeds = self.MGNN(g, x)
        # mu_init, _, _ = cluster(embeds, self.K, num_iter, init=self.init, cluster_temp=self.cluster_temp)
        mu_init = cluster_(embeds, self.K)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


def MGNN():
    model_cluster = MGNNClusterNet(nout=args.embed_dim, K=args.K, cluster_temp=args.clustertemp).to(device)
    optimizer = optim.Adam(model_cluster.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    curr_test_loss = 0
    best_train_val = 100  # 存储迄今为止最佳的训练损失值
    for t in range(args.train_iters):
        num_cluster_iter = args.num_cluster_iter
        if t >= 200:
            num_cluster_iter = 5
        mu, r, embeds, dist = model_cluster(g, x, num_cluster_iter)
        loss = loss_modularity(r, bin_adj_all, test_object, device)
        loss = -loss
        optimizer.zero_grad()
        loss.backward()
        if t % 10 == 0:
            r = torch.softmax(100 * r, dim=1)
            loss_test = loss_modularity(r, bin_adj_all, test_object, device)
            if loss.item() < best_train_val:
                best_train_val = loss.item()
                curr_test_loss = loss_test.item()
        losses.append(loss.item())
        optimizer.step()
    return curr_test_loss


if __name__ == '__main__':
    MGNN_main_auc = MGNN()
