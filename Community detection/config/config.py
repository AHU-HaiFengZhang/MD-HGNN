import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='CiteSeer')
args.add_argument('--learning_rate', type=float, default=0.001)
args.add_argument('--hidden', type=int, default=128)  # 50
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--K', type=int, default=6)
args.add_argument('--clustertemp', type=int, default=100)
args.add_argument('--train_iters', type=int, default=200)
args.add_argument('--num_cluster_iter', type=int, default=1)
args.add_argument('--embed_dim', type=int, default=64)  # 50
# args.add_argument('--model', default='gcn')
# args.add_argument('--early_stopping', type=int, default=10)
# args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--emb_type', type=str, default='attention', help='embedding type:graphwave/node2vec')
args.add_argument('--model_type', type=str, default="hgcn", help='model type')  # node_connection degree
args.add_argument('--hgcn_construct_type', type=str, default='node_connection', help='model type')
args.add_argument('--seed', type=int, default=777, help='seed')
args.add_argument('--heads', type=int, default=8, help='attention heads')
args.add_argument('--residual', type=bool, default=False, help='residual')
args.add_argument('--fusion_type', type=str, default='attention', help='attention or concat')

args = args.parse_args()
