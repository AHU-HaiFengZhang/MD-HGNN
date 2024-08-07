import torch.optim as optim
from dhg import Hypergraph
from torch_geometric.datasets import Planetoid
from model.Model import *
from utils import *
import json
from config.config import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid('data', name=args.dataset)
# Data(x=[2708, 1433], edge_index=[2, 8448], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_labe

adj = adj_generate(dataset.edge_index).numpy()
adj_all, _ = load_data_1(adj, dataset.x)

bin_adj_all = (adj_all.to_dense() > 0).float()  # standard adjacency matrix
test_object = make_modularity_matrix(bin_adj_all)
bin_adj_all = bin_adj_all.to(device)
test_object = test_object.to(device)

graph = create_graph_from_edge_list(dataset.edge_index)
num_nodes = nx.number_of_nodes(graph)

with open('data/cora/motif/cora_list_d.json', 'r') as f:  # 删除所有嵌套
    list_d = json.load(f)
hg_d = Hypergraph(num_nodes, list_d, device=device)

nfeat = dataset.x.shape[1]
dataset.x = dataset.x.to(device)
dataset.edge_index = dataset.edge_index.to(device)


def HGNNP_GCN_d():
    model_cluster = HGNNP_GCNClusterNet(nfeat=nfeat, nhid=args.hidden, nout=args.embed_dim, dropout=args.dropout,
                                        K=args.K, cluster_temp=args.clustertemp)
    model_cluster = model_cluster.to(device)
    optimizer = optim.Adam(model_cluster.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    curr_test_loss = 0
    best_train_val = 100  # 存储迄今为止最佳的训练损失值
    for t in range(args.train_iters):
        num_cluster_iter = args.num_cluster_iter
        if t >= 200:
            num_cluster_iter = 5
        mu, r, embeds, dist = model_cluster(dataset.x, dataset.edge_index, hg_d, num_cluster_iter)
        loss = loss_modularity(r, bin_adj_all, test_object)  # #   ##################
        loss = -loss
        optimizer.zero_grad()
        loss.backward()
        if t % 10 == 0:
            r = torch.softmax(100 * r, dim=1)
            loss_test = loss_modularity(r, bin_adj_all, test_object)  # #   ##################
            if loss.item() < best_train_val:
                best_train_val = loss.item()
                curr_test_loss = loss_test.item()
            log = 'Iterations: {:03d}, ClusterNet modularity: {:.4f}'
            print(log.format(t, curr_test_loss))
        losses.append(loss.item())
        optimizer.step()
    return curr_test_loss


if __name__ == '__main__':
    HGNNP_GCN_d_auc = HGNNP_GCN_d()
