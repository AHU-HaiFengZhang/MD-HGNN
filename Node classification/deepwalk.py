import networkx as nx
from gensim.models import Word2Vec
import random
from utils import *
import pickle
import numpy as np
from torch_geometric.datasets import Planetoid


class DeepWalk:
    def __init__(self, G, walk_length=80, num_walks=10, embedding_dim=7, window_size=10, workers=8):  # cora 5 4
        """
        初始化DeepWalk类
        参数：
        G (nx.Graph): 待处理的网络X图对象
        walk_length (int): 随机游走的步长（默认为80）
        num_walks (int): 每个节点开始的游走次数（默认为10）
        embedding_dim (int): 节点嵌入维度（默认为128）
        window_size (int): Word2Vec模型的窗口大小（默认为5）
        workers (int): 并行计算的进程数（默认为4）
        """
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers

    def simulate_walks(self):
        """
        生成随机游走序列
        返回：
        list[list[str]]: 所有节点游走序列组成的列表
        """
        walks = []
        nodes = list(self.G.nodes())

        for _ in range(self.num_walks):
            for node in nodes:
                walk = [str(node)]  # 将节点转换为字符串以便Word2Vec处理
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.G.neighbors(node))
                    if neighbors:
                        next_node = random.choice(neighbors)
                        walk.append(str(next_node))
                        node = next_node
                walks.append(walk)

        return walks

    def learn_embeddings(self, walks):
        """
        使用Word2Vec模型学习节点嵌入
        参数：
        walks (list[list[str]]): 随机游走序列列表
        返回：
        dict[str, np.ndarray]: 节点ID到嵌入向量的字典
        """
        model = Word2Vec(
            walks,
            size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # 使用Skip-gram模型
            workers=self.workers
        )
        if hasattr(model.wv, 'index_to_key'):
            node_embeddings = {node_id: model.wv[node_str] for node_id, node_str in enumerate(model.wv.index_to_key)}
        elif hasattr(model.wv, 'index2word'):
            node_embeddings = {node_id: model.wv[node_str] for node_id, node_str in enumerate(model.wv.index2word)}
        elif hasattr(model.wv, 'key_to_index'):
            node_embeddings = {node_id: model.wv[node_str] for node_id, node_str in
                               enumerate(model.wv.key_to_index.keys())}
        else:
            raise AttributeError("model.wv doesn't have expected attributes")

        # node_embeddings = {node_id: model.wv[node_str] for node_id, node_str in enumerate(model.wv.index_to_key)}
        return node_embeddings

    def run(self):
        """
        执行DeepWalk算法，包括随机游走和学习节点嵌入
        返回：
        dict[str, np.ndarray]: 节点ID到嵌入向量的字典
        """
        walks = self.simulate_walks()
        node_embeddings = self.learn_embeddings(walks)
        return node_embeddings


def edges_to_graph(edge_matrix):
    G = nx.Graph()
    edge_matrix = np.array(edge_matrix)
    # 添加边到图中
    for edge in edge_matrix.T:  # 转置矩阵以便迭代每条边
        G.add_edge(edge[0], edge[1])
    return G

# 示例：使用DeepWalk对给定图G进行节点嵌入学习
dataset = Planetoid('data', name='Cora')  # 就是张量
# adj = adj_generate(dataset.edge_index).numpy()  # 一个数组
# rows = np.where(np.all(adj == 0, axis=1))  # 48, 行列相同
# adj = np.delete(np.delete(adj, rows, axis=0), rows, axis=1)  # 3279,3279
# adj_ = sp.csr_matrix(adj)
# _, edge_index, _ = Get_edge_pyg(adj_)  # 边的条数没有变，但是对应的节点标号变了
graph = edges_to_graph(dataset.edge_index)
# G = nx.read_edgelist('your_graph.edgelist', create_using=nx.Graph())  # 加载网络数据
# graph = nx.Graph()
#
# with open('data/gene/gene.edges', 'r') as file:
#     for line in file:
#         line = line.strip()
#         source, target, _ = line.split(',')
#         source, target = int(source) - 1, int(target) - 1  # 让节点标号从0开始
#         graph.add_edge(source, target)
deepwalk = DeepWalk(graph)
node_embeddings = deepwalk.run()
# 保存字典到文件
with open('data/Deepwalk/cora.pkl', 'wb') as f:
    pickle.dump(node_embeddings, f)

with open('data/Deepwalk/cora.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

values_list = list(loaded_dict.values())
# 将列表转换为NumPy矩阵
matrix = np.array(values_list)

print("字典：", loaded_dict)
print("矩阵：\n", matrix)
