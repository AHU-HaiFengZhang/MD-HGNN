import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse import hstack
from motif_generetor import MotifGenerator
from motif_generetor import MotifGenerator_edge
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import degree
import warnings
from scipy.sparse.linalg import eigsh
import os
import time
import pickle
import tables as tb


def adj_generate(edge_index):
    num_nodes = edge_index.max().item() + 1
    print(num_nodes)
    # 创建邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # 将矩阵数据转换为邻接矩阵
    for col in edge_index.t():
        node1, node2 = col
        adj_matrix[node1, node2] = 1
        adj_matrix[node2, node1] = 1
    return adj_matrix


def Get_edge_pyg(A):  # 将邻接矩阵 A 转换为具有权重的边列表形式
    """
    Input: 邻接矩阵(weighted or not weighted)
    process: 转化为具有权重的邻接矩阵形式,(.row, .col为连边, values为权重)
    return: 返回pyg所需要的edge形式, 边, 边特征
  """
    edge_index_temp = sp.coo_matrix(A)
    values = edge_index_temp.data  # 对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    # edge_index_A = torch.LongTensor(indices)
    edge = torch.LongTensor(indices)
    edge_attribute = torch.FloatTensor(values)
    edge_index = torch.sparse_coo_tensor(edge, edge_attribute, edge_index_temp.shape)
    return edge_index, edge, edge_attribute


def merge_sparse_matrices(matrix1, matrix2):
    columns_to_remove = []
    for col1 in range(matrix1.shape[1]):
        is_contained = False
        for col2 in range(matrix2.shape[1]):
            if (matrix1.getcol(col1).nnz > 0 and
                    matrix2.getcol(col2).nnz > 0 and
                    (matrix1.getcol(col1).multiply(matrix2.getcol(col2)).nnz == matrix1.getcol(col1).nnz)):
                is_contained = True
                break
        if is_contained:
            columns_to_remove.append(col1)

    matrix1 = matrix1[:, [col for col in range(matrix1.shape[1]) if col not in columns_to_remove]]
    merged_matrix = sp.hstack((matrix1, matrix2), format='csr')
    return merged_matrix


def merge_lists(*lists):
    merged_list = [item for sublist in lists for item in sublist]
    return merged_list


def generate_h_278_d(A):
    Mo = MotifGenerator()
    A = np.array(A)
    l2 = Mo.M32_generator(A)
    l7 = Mo.M45_generator(A)
    l8 = Mo.M46_generator(A)
    nested_list = merge_lists(l2, l7, l8)
    final_list = []

    for sublist in nested_list:
        is_subset = False
        sublist_elements = set(sublist[0].split())  # 拆分当前子列表的元素为单独的数字
        for other_sublist in nested_list:
            if sublist != other_sublist:
                other_sublist_elements = set(other_sublist[0].split())  # 拆分其他子列表的元素为单独的数字
                if sublist_elements.issubset(other_sublist_elements):
                    is_subset = True
                    break
        if not is_subset:
            final_list.append(sublist)

    # print("原始列表:")
    # print(len(nested_list), nested_list)
    # print("\n删除子列表的元素被其他子列表包含的子列表后的列表:")
    # print(len(final_list), final_list)
    num_vertices = A.shape[0]
    num_edges = int(np.sum(A) / 2)  # 无向图边数为邻接矩阵元素和的一半

    # 初始化关联矩阵
    sparse_matrix_1 = lil_matrix((num_vertices, num_edges), dtype=np.float64)
    # 遍历邻接矩阵，生成关联矩阵
    edge_idx = 0
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if A[i][j] == 1:
                sparse_matrix_1[i, edge_idx] = 1
                sparse_matrix_1[j, edge_idx] = 1
                edge_idx += 1
    # print('inc_matrix:', sparse_matrix_1.shape)
    sparse_matrix_1 = sparse_matrix_1.tocsr()
    N = np.shape(sparse_matrix_1)[0]
    # matrix = np.random.randint(0, 1, size=(N, len(merge_edge)))
    sparse_matrix_2 = lil_matrix((N, len(final_list)), dtype=np.float64)
    index = 0
    for each in final_list:
        edge_list = list(map(int, each[0].split(" ")))
        for edge in edge_list:
            sparse_matrix_2[edge, index] = 1
        index += 1  # 按列进行赋值
    sparse_matrix_2 = sparse_matrix_2.tocsr()
    H = merge_sparse_matrices(sparse_matrix_1, sparse_matrix_2)
    print(H.shape, type(H))
    return H


def generate_edge_degree(A):
    MO_ = MotifGenerator_edge
    A32 = MO_.M32_edge_degree(A)
    A45 = MO_.M45_edge_degree(A)
    A46 = MO_.M46_edge_degree(A)
    # print(A32.toarray())
    # print(A45.toarray())
    # print(A46.toarray())
    A_all = A32 + A45 + A46
    return A_all  # 输出结果不包含原始邻接矩阵


def find_true_indices(bool_list):
    return [index for index, value in enumerate(bool_list) if value]


def merge_matrices(matrix1, matrix2):
    # merged_matrix = np.concatenate((matrix1, matrix2), axis=1)  # 合并矩阵1和矩阵2

    # 检查每一列
    columns_to_remove = []
    for col1 in range(matrix1.shape[1]):
        is_contained = False
        for col2 in range(matrix2.shape[1]):
            if np.count_nonzero(matrix1[:, col1]) > 0 and np.all(
                    matrix1[:, col1][matrix1[:, col1] != 0] == matrix2[:, col2][matrix1[:, col1] != 0]):
                is_contained = True
                break
        if is_contained:
            columns_to_remove.append(col1)
            # print(columns_to_remove)
            # print(len(columns_to_remove))

    # 删除需要删除的列
    matrix1 = np.delete(matrix1, columns_to_remove, axis=1)
    merged_matrix = np.concatenate((matrix1, matrix2), axis=1)
    print(columns_to_remove)
    print(len(columns_to_remove))
    return merged_matrix


def get_X_matrix(file):
    data = pd.read_csv(file, skiprows=1, delimiter=' ', header=None, dtype=float)
    data = data.sort_values(by=0)
    x = data.iloc[:, 1:]
    # row_num = x.shape[0]
    # col_num = x.shape[1]
    x = torch.from_numpy(x.values)
    x = torch.as_tensor(x, dtype=torch.float32)
    return x


# 损失函数
# def loss_modularity(mu, r, embeds, dist, bin_adj, mod, args):
def loss_modularity(r, bin_adj, mod):
    bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()


# 获得模块度矩阵
def make_modularity_matrix(adj):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))  # 邻接矩阵 adj 的对角线元素设置为0
    degrees = adj.sum(dim=0).unsqueeze(1)  # 节点度的列向量
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def mx_to_sparse_tensor(mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data_1(adj_matrix, node_features):
    """Load network (graph) given adjacency matrix and node features"""
    adj = sp.coo_matrix(adj_matrix)
    adj = normalize(adj + sp.eye(adj_matrix.shape[0]))
    adj = mx_to_sparse_tensor(adj)
    features = torch.tensor(node_features).to_sparse()
    return adj, features


def create_graph_from_edge_list(edge_list):
    G = nx.Graph()
    for col in edge_list.t().numpy():
        u, v = col
        G.add_edge(u, v)
    return G

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def get_X_matrix1(file, emb_type):
    if emb_type == 'graphwave':
        data = pd.read_csv(file, delimiter=',', dtype=float)
    if emb_type == 'node2vec' or emb_type == 'struc2vec' or emb_type == 'deepwalk':
        data = pd.read_csv(file, skiprows=1, delimiter=' ', header=None, dtype=float)
        data = data.sort_values(by=0)
    x = data.iloc[:, 1:]
    row_num = x.shape[0]
    col_num = x.shape[1]
    x = torch.from_numpy(x.values)
    x = torch.as_tensor(x, dtype=torch.float32)
    return x, row_num, col_num


def construct_hypergraph(edge_index, num_nodes, construct_type, args):  # ,args
    if construct_type == 'node_connection':
        hypergraph_list = list()
        now_src_node = -1
        for i in tqdm(range(len(edge_index[0]))):
            src_node = edge_index[0][i]
            if src_node != now_src_node:
                hypergraph_list.append([0] * num_nodes)
                now_src_node = src_node
                now_dst_node = edge_index[1][i]
                hypergraph_list[-1][now_dst_node] = 1
            else:
                now_dst_node = edge_index[1][i]
                hypergraph_list[-1][now_dst_node] = 1
        hypergraph_matrix = torch.Tensor(hypergraph_list).t()
        node_set = list()
        edge_set = list()
        for i in tqdm(range(hypergraph_matrix.shape[1])):
            for j in range(hypergraph_matrix.shape[0]):
                if hypergraph_matrix[j][i] == 1:
                    node_set.append(j)
                    edge_set.append(i)
        hyperedge_index = [node_set, edge_set]
    if construct_type == 'degree':
        degree_dict = dict()
        degree_list = degree(edge_index[0]).tolist()
        for idx, value in enumerate(degree_list):
            value = int(value)
            if not degree_dict.get(value):
                degree_dict[value] = [idx]
            else:
                degree_dict[value].append(idx)
        node_set = list()
        edge_set = list()
        for idx, key in enumerate(degree_dict):
            node_set.extend(degree_dict[key])
            edge_set.extend([idx] * len(degree_dict[key]))
        hyperedge_index = [node_set, edge_set]
    hyperedge_index = torch.as_tensor(hyperedge_index, dtype=torch.long)
    torch.save(hyperedge_index,f'data/hyperedge_index_{args.dataset}_{args.hgcn_construct_type}.pt')
    return hyperedge_index


def normalize_adj(sp_mat):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rowsum = np.array(np.sum(sp_mat, axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).reshape([-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        support = sp_mat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        I = sp.eye(sp_mat.shape[0])
        L = I - support
        L = L - ((lmax(L) / 2) * I)

    return L


class MotifCounter:
    def __init__(self, dataset_name, simple_relational_digraphs, cache_dir, logger=None):
        self.dataset_name = dataset_name
        self.target_motifs = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13')
        self.logger = logger
        self.cache_dir = cache_dir
        self.num_node = simple_relational_digraphs[0].shape[0]
        self.num_edge = simple_relational_digraphs[0].nnz
        row, col = simple_relational_digraphs[0].nonzero()
        self.edge_index = list(zip(row.tolist(), col.tolist()))
        ckg_adj = sum(simple_relational_digraphs)
        ckg_bi = ckg_adj.multiply(ckg_adj.transpose())
        self.ckg_csr = sp.csr_matrix(ckg_adj) + sp.eye(self.num_node)
        self.ckg_csc = sp.csc_matrix(ckg_adj)
        self.ckg_bi = ckg_bi.tocsr()
        self.raw_graph = ckg_adj.tocoo()
        self.raw_bi_graph = ckg_bi.tocoo()

        self.ckg_bi = ckg_bi.tocsr()
        src, dst = ckg_adj.nonzero()
        self.edges_from_raw_graph = list(zip(src.tolist(), dst.tolist()))
        src, dst = ckg_bi.nonzero()
        self.bi_edges_from_raw_graph = list(zip(src.tolist(), dst.tolist()))

    def split_13motif_adjs(self):
        cache_path = os.path.join(self.cache_dir, f'motif_adj4{self.dataset_name}')
        check_dir(cache_path)
        motif_adj_file = 'm{}_adj4' + self.dataset_name + '.pkl'
        motif_adjs = []
        if os.path.exists(os.path.join(cache_path, motif_adj_file.format(1))):
            for i in range(13):
                motif_adjs.append(load(os.path.join(cache_path, motif_adj_file.format(i+1))))  # list of csr
        else:
            t = time.time()
            # edge_motif_weight = torch.zeros(train_graph.number_of_edges(), len(self.target_motifs))
            U = self.raw_graph
            B = self.raw_bi_graph

            for i, motif_name in enumerate(self.target_motifs):
                print(f"Begin count {motif_name}-motif weight for each edge...")
                if motif_name == 'M1':
                    if self.dataset_name in ['amazon-book']:
                        if os.path.exists(self.cache_dir + 'uuut.h5'):
                            fr = tb.open_file(self.cache_dir + 'uuut.h5', 'r')
                            C = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()
                        else:
                            C = tb_matmul_and_multiply(U, U, U.transpose(), self.cache_dir + 'uuut.h5')
                    else:
                        C = U.dot(U).multiply(U.transpose())
                    # C = U.dot(U).multiply(U.transpose())
                    motif_adj = C + C.transpose()
                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)

                elif motif_name == 'M2':
                    if self.dataset_name in ['amazon-book']:
                        if os.path.exists(self.cache_dir + 'buut.h5'):
                            fr = tb.open_file(self.cache_dir + 'buut.h5', 'r')
                            C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'ubut.h5', 'r')
                            C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'uub.h5', 'r')
                            C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            C = C1 + C2 + C3
                        else:
                            C1 = tb_matmul_and_multiply(B, U, U.transpose(), self.cache_dir + 'buut.h5')
                            C2 = tb_matmul_and_multiply(U, B, U.transpose(), self.cache_dir + 'ubut.h5')
                            C3 = tb_matmul_and_multiply(U, U, B, self.cache_dir + 'uub.h5')

                            C = C1 + C2 + C3
                    else:
                        C = B.dot(U).multiply(U.transpose()) + U.dot(B).multiply(U.transpose()) + U.dot(U).multiply(B)

                    motif_adj = C + C.transpose()
                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                elif motif_name == 'M3':
                    if self.dataset_name in ['amazon-book']:
                        if os.path.exists(self.cache_dir + 'bbu.h5'):
                            fr = tb.open_file(self.cache_dir + 'bbu.h5', 'r')
                            C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'bub.h5', 'r')
                            C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'ubb.h5', 'r')
                            C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            C = C1 + C2 + C3
                        else:
                            C1 = tb_matmul_and_multiply(B, B, U, self.cache_dir + 'bbu.h5')
                            C2 = tb_matmul_and_multiply(B, U, B, self.cache_dir + 'bub.h5')
                            C3 = tb_matmul_and_multiply(U, B, B, self.cache_dir + 'ubb.h5')

                            C = C1 + C2 + C3
                    else:
                        C = B.dot(B).multiply(U) + B.dot(U).multiply(B) + U.dot(B).multiply(B)
                    motif_adj = C + C.transpose()
                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                elif motif_name == 'M4':
                    if self.dataset_name in ['amazon-book']:
                        if os.path.exists(self.cache_dir + 'bbb.h5'):
                            fr = tb.open_file(self.cache_dir + 'bbb.h5', 'r')
                            motif_adj = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()
                        else:
                            motif_adj = tb_matmul_and_multiply(B, B, B, self.cache_dir + 'bbb.h5')
                    else:
                        motif_adj = B.dot(B).multiply(B)

                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                elif motif_name == 'M5':
                    # C = U.dot(U).multiply(U) + U.dot(U.transpose()).multiply(U) + U.transpose().dot(U).multiply(U)
                    # C1 = U.dot(U).multiply(U) + U.transpose().dot(U).multiply(U)
                    if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b']:
                        if os.path.exists(self.cache_dir + 'uuu.h5'):
                            fr = tb.open_file(self.cache_dir + 'uuu.h5', 'r')
                            C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'uutu.h5', 'r')
                            C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'utuu.h5', 'r')
                            C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            C = C1 + C2 + C3
                        else:
                            C1 = tb_matmul_and_multiply(U, U, U, self.cache_dir + 'uuu.h5')
                            C2 = tb_matmul_and_multiply(U, U.transpose(), U, self.cache_dir + 'uutu.h5')
                            C3 = tb_matmul_and_multiply(U.transpose(), U, U, self.cache_dir + 'utuu.h5')

                            C = C1 + C2 + C3
                    else:
                        C = U.dot(U).multiply(U) + U.dot(U.transpose()).multiply(U) + U.transpose().dot(U).multiply(U)

                    motif_adj = C + C.transpose()
                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                elif motif_name == 'M6':
                    # U = convert_sparse_matrix_to_sparse_tensor(U).to(device)
                    # B = convert_sparse_matrix_to_sparse_tensor(B).to(device)
                    # motif_adj = U.dot(B) * U + B.dot(U.t()) * U.t() + U.t().dot(U) * B
                    if self.dataset_name in ['amazon-book', 'yelp2018']:
                        if os.path.exists(self.cache_dir + 'ubu.h5'):
                            fr = tb.open_file(self.cache_dir + 'ubu.h5', 'r')
                            C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'butut.h5', 'r')
                            C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'utub.h5', 'r')
                            C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            motif_adj = C1 + C2 + C3
                        else:
                            C1 = tb_matmul_and_multiply(U, B, U, self.cache_dir + 'ubu.h5')
                            C2 = tb_matmul_and_multiply(B, U.transpose(), U.transpose(), self.cache_dir + 'butut.h5')
                            C3 = tb_matmul_and_multiply(U.transpose(), U, B, self.cache_dir + 'utub.h5')
                            motif_adj = C1 + C2 + C3
                    else:
                        motif_adj = U.dot(B).multiply(U) + B.dot(U.transpose()).multiply(U.transpose()) + U.transpose().dot(U).multiply(B)

                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                elif motif_name == 'M7':
                    # C1 = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U)
                    if self.dataset_name in ['amazon-book', 'yelp2018', 'lfm1b']:
                        if os.path.exists(self.cache_dir + 'utbut.h5'):
                            fr = tb.open_file(self.cache_dir + 'utbut.h5', 'r')
                            C1 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'buu.h5', 'r')
                            C2 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            fr = tb.open_file(self.cache_dir + 'uutb.h5', 'r')
                            C3 = sp.coo_matrix((fr.root.data[:], (fr.root.row[:], fr.root.col[:])), shape=U.shape)
                            fr.close()

                            motif_adj = C1 + C2 + C3
                        else:
                            C1 = tb_matmul_and_multiply(U.transpose(), B, U.transpose(), self.cache_dir + 'utbut.h5')
                            C2 = tb_matmul_and_multiply(B, U, U, self.cache_dir + 'buu.h5')
                            C3 = tb_matmul_and_multiply(U, U.transpose(), B, self.cache_dir + 'uutb.h5')
                            motif_adj = C1 + C2 + C3
                    else:
                        motif_adj = U.transpose().dot(B).multiply(U.transpose()) + B.dot(U).multiply(U) + U.dot(U.transpose()).multiply(B)

                    # save(cache_path + motif_adj_file.format(i+1), motif_adj)
                    # neighbors, center_nodes = motif_adj.nonzero()
                    # edge_motif_weight[train_graph.edge_ids(neighbors, center_nodes),
                    #                   self.target_motifs.index(motif_name)] += torch.tensor(motif_adj.data)

                else:  # 处理开motif
                    motif_adj = sp.csr_matrix((self.num_node, self.num_node))
                    for center_node in tqdm(range(self.num_node)):
                        # center_node = 95595
                        # print(f"center_node: {center_node}")
                        center_indices, out_neighbors = self.ckg_csr[center_node].nonzero()  # 95595
                        # 此处的out_neighbors和in_neighbors都是包含双向邻居的
                        out_neighbors = out_neighbors.tolist()  # 带自环，闭合motif专用

                        in_neighbors, center_indices = self.ckg_csc[:, center_node].nonzero()
                        in_neighbors = in_neighbors.tolist()

                        # 过滤后的out,in，还要再和out做交集，把合并的out和in再分类拿回来
                        # 问题是，合并之前，out和in会有交集的，这是否有问题？

                        center_indices, bi_neighbors = self.ckg_bi[center_node].nonzero()  # 双向边
                        bi_neighbors = bi_neighbors.tolist()

                        pure_in_neighbors = list(set(in_neighbors) - set(bi_neighbors))  # 纯入邻居
                        pure_out_neighbors = list(set(out_neighbors) - set(bi_neighbors))  # 纯出邻居

                        if motif_name == 'M8':  # 中间角度，开
                            if len(out_neighbors) > 0:
                                out_weights = len(out_neighbors) - 1
                                # edge_motif_weight[train_graph.edge_ids(out_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += out_weights

                                motif_adj[out_neighbors, [center_node]] += out_weights

                        elif motif_name == 'M9':  # 中间角度，开
                            # 一旦两边都有相同的邻居，就要分pure和bi
                            # 开Motif运算双向边和自己组合，不必-2
                            # bn_weights = len(out_neighbors_wc) + len(in_neighbors) - 2  # 双向邻居的权重
                            # 因为双向邻居出和入组合，入和出组合是相同情况，重复了一次，刚好达到次数x2
                            if len(bi_neighbors) > 0:
                                bi_weights = len(out_neighbors) + len(in_neighbors)  # 双向邻居的权重
                                # edge_motif_weight[train_graph.edge_ids(bi_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += bi_weights

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_out_neighbors) > 0:
                                pure_out_weights = len(in_neighbors)  # 纯出邻居的权重
                                # edge_motif_weight[train_graph.edge_ids(pure_out_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += pure_out_weights

                                motif_adj[pure_out_neighbors, [center_node]] += pure_out_weights

                            if len(pure_in_neighbors) > 0:
                                pure_in_weights = len(out_neighbors)  # 纯入邻居的权重
                                # edge_motif_weight[train_graph.edge_ids(pure_in_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += pure_in_weights

                                motif_adj[pure_in_neighbors, [center_node]] += pure_in_weights

                        elif motif_name == 'M10':  # 中间角度，开
                            if len(in_neighbors) > 0:
                                in_weights = len(in_neighbors) - 1
                                # edge_motif_weight[train_graph.edge_ids(in_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += in_weights

                                motif_adj[in_neighbors, [center_node]] += in_weights

                        elif motif_name == 'M11':  # 中间角度，开
                            # 邻居作为双向邻居时可以和所有出邻居组合，除了和该邻居相同的，所以-1
                            # 当作为出邻居时，可以和所有的双向邻居组合，但不能和该邻居相同，所以-1
                            if len(bi_neighbors) > 0:
                                # bi_weights = len(out_neighbors) + len(bi_neighbors) - 2  # 双向邻居的权重
                                bi_weights = len(out_neighbors) - 1  # 双向邻居的权重

                                # edge_motif_weight[train_graph.edge_ids(bi_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += bi_weights

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_out_neighbors) > 0:
                                pure_out_weights = len(bi_neighbors)  # 纯出邻居的权重，从双向产生的出邻居不要
                                # edge_motif_weight[train_graph.edge_ids(pure_out_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += pure_out_weights

                                motif_adj[pure_out_neighbors, [center_node]] += pure_out_weights

                        elif motif_name == 'M12':  # 中间角度，开
                            if len(bi_neighbors) > 0:
                                # bi_weights = len(bi_neighbors) + len(in_neighbors) - 2
                                bi_weights = len(in_neighbors) - 1
                                # edge_motif_weight[train_graph.edge_ids(bi_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += bi_weights

                                motif_adj[bi_neighbors, [center_node]] += bi_weights

                            if len(pure_in_neighbors) > 0:
                                pure_in_weights = len(bi_neighbors)
                                # edge_motif_weight[train_graph.edge_ids(pure_in_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += pure_in_weights

                                motif_adj[pure_in_neighbors, [center_node]] += pure_in_weights

                        elif motif_name == 'M13':  # 中间角度，开
                            if len(bi_neighbors) > 0:
                                bi_weights = len(bi_neighbors) - 1
                                # edge_motif_weight[train_graph.edge_ids(bi_neighbors, center_node),
                                #                   self.target_motifs.index(motif_name)] += bi_weights

                                motif_adj[bi_neighbors, [center_node]] += bi_weights
                    motif_adj = motif_adj + motif_adj.transpose()

                save(os.path.join(cache_path, motif_adj_file.format(i + 1)), motif_adj)
                motif_adjs.append(motif_adj)

            print(f"The {len(self.target_motifs)} motif weight count process took {time.time() - t:.2f}s.")
            # torch.save(edge_motif_weight, self.cache_dir + edge_weight_file)

        edge_weight_sum = []
        for motif_adj in motif_adjs:
            edge_weight_sum.append(motif_adj.sum())

        num_motifs = {}
        for i, motif_name in enumerate(self.target_motifs):
            if int(motif_name[1:]) < 8:
                num_motifs[motif_name] = int(edge_weight_sum[i] / 6)
                # assert edge_weight_sum[i] % 6 == 0
            else:
                num_motifs[motif_name] = int(edge_weight_sum[i] / 2)
                # assert edge_weight_sum[i] % 2 == 0

        info = [f"{motif_name}: {num_motifs[motif_name]} |" for motif_name in num_motifs]
        if self.logger is not None:
            self.logger.info(f"=====* Motifs included in {self.dataset_name} dataset *=====")
            self.logger.info(" ".join(info))
        else:
            print(f"=====* Motifs included in {self.dataset_name} dataset *=====")
            print(" ".join(info))
        del num_motifs
        # print(f"calc num motifs cost {time.time() - t:.2f}s")

        # train_graph = train_graph.to(device)
        # edge_motif_weight = edge_motif_weight.to(device)
        # train_graph.edata['motif_weight'] = edge_motif_weight

        return motif_adjs

    def get_motif_weight(self, motif_mats):
        weight_file = os.path.join(self.cache_dir, 'motif_weight.pt')
        if os.path.exists(weight_file):
            motif_weight = torch.load(weight_file)
            print("motif weight file loaded.")
        else:
            motif_weight = torch.zeros(self.num_edge, 13)
            for i, motif_mat in enumerate(motif_mats):
                data = motif_mat.data.tolist()
                for j, node_pair in enumerate(zip(*motif_mat.nonzero())):
                    eid = self.edge_index.index(node_pair)
                    motif_weight[eid, i] += data[j]

            torch.save(motif_weight, weight_file)
            print("motif weight file saved.")

        return motif_weight


def convert_sparse_matrix_to_th_sparse_tensor(sp_mat):
    sp_mat = sp_mat.tocoo()
    indices = torch.LongTensor([sp_mat.row.tolist(), sp_mat.col.tolist()])
    data = torch.FloatTensor(sp_mat.data.tolist())
    size = torch.Size(sp_mat.shape)

    return torch.sparse.FloatTensor(indices, data, size).coalesce()


def load(path):
    if 'pkl' in path:
        with open(path, 'rb') as fr:
            data = pickle.load(fr)
        return data
    elif 'npy' in path:
        return np.load(path, allow_pickle=False)
    else:
        with open(path) as fw:
            data = fw.readlines()
        return data


def save(path, data):
    if 'pkl' in path:
        with open(path, 'wb') as fw:
            pickle.dump(data, fw)
    elif 'npy' in path:
        np.save(path, data)
    else:
        with open(path, 'w') as fw:
            fw.writelines(data)


def tb_matmul_and_multiply(sp_mat1, sp_mat2, sp_mat3, hdf5_name, chunk_size=10000):
    csc_mat2 = sp_mat2.tocsc()
    csc_mat3 = sp_mat3.tocsc()
    # print(f"chunk_size={chunk_size}")
    left_dim, middle_dim, right_dim = sp_mat1.shape[0], sp_mat1.shape[1], sp_mat2.shape[1]
    fw = tb.open_file(hdf5_name, 'w')
    filters = tb.Filters(complevel=5, complib='blosc')
    row = fw.create_earray(fw.root, 'row', tb.Int32Atom(), shape=(0,), filters=filters)
    col = fw.create_earray(fw.root, 'col', tb.Int32Atom(), shape=(0,), filters=filters)
    data = fw.create_earray(fw.root, 'data', tb.Int32Atom(), shape=(0,), filters=filters)

    for i in tqdm(range(0, right_dim, chunk_size)):
        res = sp_mat1.dot(csc_mat2[:, i:min(i + chunk_size, right_dim)]).multiply(csc_mat3[:, i:min(i + chunk_size, right_dim)])
        data_i = res.data
        row_i, col_i = res.nonzero()
        data.append(data_i)
        row.append(row_i)
        col.append(i + col_i)

    product = sp.coo_matrix((fw.root.data[:], (fw.root.row[:], fw.root.col[:])), shape=sp_mat1.shape)
    fw.close()

    return product


def check_dir(directory_name):
    """

    :param directory_name: e.g. ./process_files or ../data/process_files
    :return:
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
