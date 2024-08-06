import torch.optim as optim
from dhg import Hypergraph
from dhg.models import HGNNP
from torch_geometric.datasets import Planetoid
from utils import *
from config.config import args
from model.Model import *
import json
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid('data', name=args.dataset)

# 生成无向图
adj = adj_generate(dataset.edge_index).numpy()
adj_ = sp.csr_matrix(adj)
_, edge_index, _ = Get_edge_pyg(adj_)
graph = create_graph_from_edge_list(edge_index)
num_nodes = nx.number_of_nodes(graph)

with open('data/cora/motif/cora_list_d.json', 'r') as f:
    list_d = json.load(f)
hg_d = Hypergraph(num_nodes, list_d, device=device)

n_class = int(dataset.y.max()) + 1
dataset.x = torch.Tensor(dataset.x).to(device)
edge_index = torch.Tensor(edge_index).to(device)
dataset.y = torch.Tensor(dataset.y).to(device)

idx_train = [i for i in range(140)]  # pubmed range(60),  val_mask range(60, 560)
idx_test = [i for i in range(1708, 2708)]  # pubmed range(18717, 19717)
train = torch.Tensor(idx_train).long().to(device)
test = torch.Tensor(idx_test).long().to(device)


def HGNNP_GCN_d():
    model = HGNNP_GCN(in_ch=dataset.x.shape[1], out_ch=n_class, n_hid=args.hidden, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 学习率调度器，在milestones步之后给学习率×gamma
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc, best = 0.0, 0.0
    for epoch in range(args.max_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                schedular.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            idx = train if phase == 'train' else test
            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):  # 训练集时会自动求导，测试集时关闭自动求导
                outputs = model(dataset.x, edge_index, hg_d)  # (fts, G, tensor_adjacency)
                loss = criterion(outputs[idx], dataset.y[idx])
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * dataset.x.size(0)
            running_corrects += torch.sum(preds[idx] == dataset.y.data[idx])
            epoch_acc = running_corrects.double() / len(idx)
            f1_score_average_micro = f1_score(dataset.y.data[idx], preds[idx], average='macro')
            if phase == 'test' and f1_score_average_micro > best:
                # print(f1_score_average_micro)
                best = f1_score_average_micro
                # print(best)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # print(best_acc)
    # measure_result = classification_report(y.data[idx], preds[idx])
    # print(measure_result)
    return best_acc, best


if __name__ == '__main__':
    acc, f1 = HGNNP_GCN_d()