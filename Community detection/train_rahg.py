import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
from model.model_gmr import *
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
data1 = dataset.data.to(device)
edge_index = dataset.edge_index
x_role, num_nodes, num_node_features_r = get_X_matrix1(f'data/{args.dataset}.csv', 'graphwave')
x_stru, num_node_features_s = dataset.x, dataset.x.shape[1]  # 不是特征矩阵而是单位矩阵
x_role = x_role.to(device)
x_stru = x_stru.to(device)
edge = to_dense_adj(dataset.data.edge_index).squeeze(0).to(device)
data1.x_role = torch.cat((dataset.data.x, x_role, edge), dim=1)
data1.x_stru = torch.cat((dataset.data.x, x_stru, edge), dim=1)  # 嵌入和原始特征相同
data1.num_node_features_r = num_node_features_r + dataset.num_node_features + num_nodes
data1.num_node_features_s = num_node_features_s + dataset.num_node_features + num_nodes
data1.num_nodes = num_nodes
data1.num_classes = 64
data1.edge_index = edge_index
if os.path.exists(f'data/hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt'):
    H = torch.load(f'data/hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt')
else:
    H = construct_hypergraph(data.edge_index, data.num_nodes, args.hgcn_construct_type, args)
data1.H = H.to(device)


def RAHG():
    model_cluster = RAHGClusterNet(data1, nhid=args.hidden, nout=args.embed_dim, gnn_type='hgcn', heads=8,
                                   dropout=args.dropout, K=args.K, cluster_temp=args.clustertemp)
    model_cluster = model_cluster.to(device)
    optimizer = optim.Adam(model_cluster.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    curr_test_loss = 0
    best_train_val = 100  # 存储迄今为止最佳的训练损失值
    for t in range(args.train_iters):
        num_cluster_iter = args.num_cluster_iter
        if t >= 200:
            num_cluster_iter = 5
        mu, r, embeds, dist = model_cluster(data1, args, num_cluster_iter)
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
     RAHG_auc = RAHG()
