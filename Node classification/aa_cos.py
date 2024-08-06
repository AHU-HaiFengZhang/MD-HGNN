from utils import *
from torch_geometric.datasets import Planetoid
from config.config import args
import json
from itertools import combinations
import random


def get_combinations(lists, seed=42):
    # if seed is not None:
    #     random.seed(seed)  # 设置随机数种子
    # random.shuffle(lists)  # 随机打乱
    # lists = lists[:(len(lists)//5)]
    result = []
    for sublist in lists:
        numbers = list(map(int, sublist[0].split()))
        comb = list(combinations(numbers, 2))
        result.extend(comb)
        result = list(set(result))
    return result


def get_cosine_similarity(lists, adj_m, matrix):
    li = []
    for sublist in lists:
        row, col = sublist
        if adj_m[row, col] != 1:
            li.append(sublist)
    r = 0
    for sublist in li:
        v1, v2 = sublist
        similarity = np.dot(matrix[v1], matrix[v2]) / (np.linalg.norm(matrix[v1]) * np.linalg.norm(matrix[v2]))
        r += similarity
    r = r / len(li)
    return r


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid('data', name=args.dataset)  # 就是张量
adj = adj_generate(dataset.edge_index).numpy()  # 一个数组
# rows = np.where(np.all(adj == 0, axis=1))  # 48, 行列相同
# adj = np.delete(np.delete(adj, rows, axis=0), rows, axis=1)  # 3279,3279
x = get_X_matrix(file=f'data_2/{args.dataset}.emb')

with open(f'data_2/{args.data}/{args.data}_l1.json', 'r') as f:
    l1 = json.load(f)
with open(f'data_2/{args.data}/{args.data}_l3.json', 'r') as f:
    l3 = json.load(f)
with open(f'data_2/{args.data}/{args.data}_l4.json', 'r') as f:
    l4 = json.load(f)
with open(f'data_2/{args.data}/{args.data}_l5.json', 'r') as f:
    l5 = json.load(f)
with open(f'data_2/{args.data}/{args.data}_l6.json', 'r') as f:
    l6 = json.load(f)
with open(f'data_2/{args.data}/{args.data}_l7.json', 'r') as f:
    l7 = json.load(f)
# l1 = get_combinations(l1)
# l3 = get_combinations(l3)
# l4 = get_combinations(l4)
# l5 = get_combinations(l5)
# l6 = get_combinations(l6)
l7 = get_combinations(l7)
# a1 = get_cosine_similarity(l1, adj, x)
# a3 = get_cosine_similarity(l3, adj, x)
# a4 = get_cosine_similarity(l4, adj, x)
# a5 = get_cosine_similarity(l5, adj, x)
# a6 = get_cosine_similarity(l6, adj, x)
a7 = get_cosine_similarity(l7, adj, x)
# print(a1, a3, a4, a5, a6, a7)
print(a7)