from torch_geometric.datasets import Planetoid
from utils import *
from config.config import args
import os
from dhg.utils import split_by_num
import pickle

dataset = Planetoid('data', name=args.dataset)  # 就是张量

# 生成无向图
graph = create_graph_from_edge_list(dataset.edge_index)
num_nodes = nx.number_of_nodes(graph)


def generate_two_lists():
    a, b = split_by_num(num_nodes, dataset.y, train_num=5)
    idx_train = find_true_indices(a)
    idx_test = find_true_indices(b)
    return idx_train, idx_test


results = []
for _ in range(20):
    result = generate_two_lists()
    results.append(result)

save_dir = 'data_1/data_split'  # 要改
save_path = os.path.join(save_dir, 'pubmed_data_split_20_2.pkl')
# 创建文件夹路径（如果不存在）
os.makedirs(save_dir, exist_ok=True)
# 使用pickle将数据保存到文件
with open(save_path, 'wb') as f:
    pickle.dump(results, f)

# load_path = 'data_1/data_split/cora_data_split_20.pkl'
# # 使用pickle从文件中加载数据
# with open(load_path, 'rb') as f:
#     results = pickle.load(f)

# 打印加载的结果
for i, (list1, list2) in enumerate(results, 1):
    print(f"Iteration {i - 1}: List 1: {list1}, List 2: {list2}")
