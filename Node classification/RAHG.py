import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from layer import *
from config.config import args
from model import *
from utils import *
import os
from torch_geometric.data import Data
import os.path as osp
from sklearn.metrics import f1_score
import warnings


warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid('data', name=args.dataset)  # 就是张量

dataset.x = torch.Tensor(dataset.x).to(device)
dataset.edge_index = torch.Tensor(dataset.edge_index).to(device)
dataset.y = torch.Tensor(dataset.y).to(device)
n_class = int(dataset.y.max()) + 1
idx_train = [i for i in range(140)]  # 原论文用60%的训练集  140
idx_test = [i for i in range(1708, 2708)]
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)

data = Data(x=dataset.x, edge_index=dataset.edge_index).to(device)
data1 = dataset.data.to(device)
edge_index = dataset.edge_index
x_role, num_nodes, num_node_features_r = get_X_matrix(f'data/{args.dataset}.csv', 'graphwave')
x_stru, num_node_features_s = dataset.x, dataset.x.shape[1]  # 不是特征矩阵而是单位矩阵
x_role = x_role.to(device)
x_stru = x_stru.to(device)
edge = to_dense_adj(dataset.data.edge_index).squeeze(0).to(device)
data1.x_role = torch.cat((dataset.data.x, x_role ,edge), dim=1)
data1.x_stru = torch.cat((dataset.data.x,  x_stru,edge), dim=1)  # 嵌入和原始特征相同
data1.num_node_features_r = num_node_features_r + dataset.num_node_features + num_nodes
data1.num_node_features_s = num_node_features_s + dataset.num_node_features + num_nodes
data1.num_nodes = num_nodes
data1.num_classes = n_class
data1.edge_index = edge_index
if os.path.exists(f'data/hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt'):
    H=torch.load(f'data/hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt')
else:
    H = construct_hypergraph(data.edge_index, data.num_nodes, args.hgcn_construct_type,args)
data1.H = H.to(device)


def RAHG_main(train, test):
    print(f"class number: {n_class}")
    model = AttentionModel(data1, args.hidden, args.model_type, args.heads, dropout=args.dropout).to(device)
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
                outputs = model(data1, args)
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
                print(f1_score_average_micro)
                best = f1_score_average_micro
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print(best_acc)
    return best_acc, best


if __name__ == '__main__':
    RAHG_main_auc, f1_rahg = RAHG_main(idx_train, idx_test)
