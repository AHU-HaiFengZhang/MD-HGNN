import torch.optim as optim
from torch_geometric.datasets import Planetoid
from config.config import args
# from MGNN_main.MGNN_Node.main import *
from torch_geometric.data import Data
import os.path as osp
from layer import *
from sklearn.metrics import classification_report

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_s = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'NodeMGNN_DATA')
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid('data', name=args.dataset)  # 就是张量

data = dataset[0]
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
hidden_dim2 = dataset.num_classes
row, col = edge_index.tolist()
sp_mat = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))  # 去除自环的邻接矩阵
edge_weight_norm = normalize_adj(sp_mat).data.reshape([-1, 1])  # 归一化之后边权重列向量

mc = MotifCounter(dataset_s, [sp_mat], osp.join(path, dataset_s, 'processed'))
motif_mats = mc.split_13motif_adjs()
motif_mats = [convert_sparse_matrix_to_th_sparse_tensor(normalize_adj(motif_mat)).to(device) for motif_mat in
              motif_mats]

weight_index_data = np.array([range(num_filter)], dtype=np.int32).repeat(num_edge, axis=0)

rel_type = [str(rel) for rel in set(weight_index_data.flatten().tolist())]
graph_data = {('P', rel, 'P'): [[], []] for rel in rel_type}
edge_data = {rel: [] for rel in rel_type}

for rel in rel_type:  # 遍历每条边及过滤索引的复杂度  2m*num_filter   构建异构图的复杂度
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


n_class = int(dataset.y.max()) + 1

idx_train = [i for i in range(140)]  # 原论文用60%的训练集  140
idx_test = [i for i in range(1708, 2708)]
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


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


def mgnn_main(train, test):
    print(f"class number: {n_class}")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 学习率调度器，在milestones步之后给学习率×gamma
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(args.max_epochs):
        if epoch % 20 == 0:
            print('-' * 10, f'Epoch {epoch}/{args.max_epochs - 1}')
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
                outputs = model(g, x)
                loss = criterion(outputs[idx], dataset.y[idx])
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * dataset.x.size(0)
            running_corrects += torch.sum(preds[idx] == dataset.y.data[idx])
            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            if epoch % 20 == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
        if epoch % 20 == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)
    print(f'Best val Acc: {best_acc:4f}')
    # measure_result = classification_report(y.data[idx], preds[idx])
    # print(measure_result)
    return best_acc
