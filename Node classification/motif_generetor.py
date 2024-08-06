import numpy as np
from scipy.sparse import lil_matrix


class MotifGenerator:
    def __init__(self):
        pass

    @staticmethod
    def M31_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            for j in range(len(lj1)):
                if A[lj1[j], b2[i]] == 0:
                    M1 = np.array([b1[i], b2[i], lj1[j]]).reshape(-1, 1)
                    M2 = np.sort(M1, axis=0)
                    M.append(M2)
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])
            for k in range(len(lj2)):
                if A[lj2[k], b1[i]] == 0:
                    M1 = np.array([b2[i], b1[i], lj2[k]]).reshape(-1, 1)
                    M2 = np.sort(M1, axis=0)
                    M.append(M2)

        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M32_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            for j in range(len(lj1)):
                if A[lj1[j], b2[i]] == 1:
                    M1 = np.array([b1[i], b2[i], lj1[j]]).reshape(-1, 1)
                    M2 = np.sort(M1, axis=0)
                    M.append(M2)

        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M41_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 0 and A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 0:
                        M1 = np.array([b1[i], b2[i], lj1[j], lj2[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M42_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1) - 1):
                for k in range(j + 1, len(lj1)):
                    if lj1[j] != lj1[k] and A[lj1[j], lj1[k]] == 0 and A[lj1[j], b2[i]] == 0 and A[lj1[k], b2[i]] == 0:
                        M1 = np.array([b1[i], b2[i], lj1[j], lj1[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
            for m in range(len(lj2) - 1):
                for n in range(m + 1, len(lj2)):
                    if lj2[m] != lj2[n] and A[lj2[m], lj2[n]] == 0 and A[lj2[m], b1[i]] == 0 and A[lj2[n], b1[i]] == 0:
                        M1 = np.array([b1[i], b2[i], lj2[m], lj2[n]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)

        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M43_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 0:
                        M1 = np.array([b1[i], b2[i], lj1[j], lj2[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M44_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 0 and (
                            (A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 0) or (
                            A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 1)):
                        M1 = np.array([b1[i], b2[i], lj1[j], lj2[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M45_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and (
                            (A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 0) or
                            (A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 1)):
                        M1 = np.array([b1[i], b2[i], lj1[j], lj2[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list

    @staticmethod
    def M46_generator(A):
        b1, b2 = np.where(np.tril(A))
        M = []

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 1:
                        M1 = np.array([b1[i], b2[i], lj1[j], lj2[k]]).reshape(-1, 1)
                        M2 = np.sort(M1, axis=0)
                        M.append(M2)
        if len(M) > 0:
            Motifs = np.unique(np.concatenate(M, axis=1).T, axis=0)  # M如果M中没有元素会报错所以修改
        else:
            Motifs = np.empty((0, 4))
        converted_list = [[' '.join(str(num) for num in sublist)] for sublist in Motifs]
        return converted_list


class MotifGenerator_edge:
    def __init__(self):
        pass

    @staticmethod
    def M31_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)
        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            for j in range(len(lj1)):
                if A[lj1[j], b2[i]] == 0:
                    W[b1[i], b2[i]] += 1
                    W[lj1[j], b2[i]] += 0.5

            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])
            for k in range(len(lj2)):
                if A[lj2[k], b1[i]] == 0:
                    W[b1[i], b2[i]] += 1
                    W[lj2[k], b1[i]] += 0.5

        W = W + W.transpose()
        return W  # W.toarray()  返回值是稀疏矩阵

    @staticmethod
    def M32_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=int)
        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            for j in range(len(lj1)):
                if A[lj1[j], b2[i]] == 1:
                    W[b1[i], b2[i]] += 1

        W = W + W.transpose()
        return W

    @staticmethod
    def M41_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)

        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):  # 遍历每一条边
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):  # 遍历lj1中的每个节点
                for k in range(len(lj2)):  # 遍历lj2中的每个节点
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 0 and A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 0:
                        # 如果lj1[j]和lj2[k]不相等，并且它们之间没有边（值为0），以及其他特定条件满足
                        W[b1[i], b2[i]] += 1  # 边(b1[i], b2[i])的度数加1
                        W[lj1[j], b1[i]] += 1
                        W[lj2[k], b2[i]] += 1
                        W[lj1[j], b2[i]] += 1
                        W[lj2[k], b1[i]] += 1
                        W[lj1[j], lj2[k]] += 1

        W = W + W.transpose()
        return W

    @staticmethod
    def M42_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)

        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1) - 1):
                for k in range(j + 1, len(lj1)):
                    if lj1[j] != lj1[k] and A[lj1[j], lj1[k]] == 0 and A[lj1[j], b2[i]] == 0 and A[lj1[k], b2[i]] == 0:
                        W[b1[i], b2[i]] += 1
                        W[lj1[j], lj1[k]] += 1 / 3
                        W[lj1[j], b2[i]] += 1 / 3
                        W[lj1[k], b2[i]] += 1 / 3

            for m in range(len(lj2) - 1):
                for n in range(m + 1, len(lj2)):
                    if lj2[m] != lj2[n] and A[lj2[m], lj2[n]] == 0 and A[lj2[m], b1[i]] == 0 and A[lj2[n], b1[i]] == 0:
                        W[b1[i], b2[i]] += 1
                        W[lj2[m], lj2[n]] += 1 / 3
                        W[lj2[m], b1[i]] += 1 / 3
                        W[lj2[n], b1[i]] += 1 / 3

        W = W + W.transpose()
        return W

    @staticmethod
    def M43_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)

        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 0:
                        W[b1[i], b2[i]] += 1
                        W[lj1[j], b2[i]] += 0.25
                        W[lj2[k], b1[i]] += 0.25

        W = W + W.transpose()
        return W

    @staticmethod
    def M44_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)
        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 0 and (
                            (A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 0) or (
                            A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 1)):
                        W[b1[i], b2[i]] += 0.5
                        W[lj1[j], b1[i]] += 0.5
                        W[lj2[k], b2[i]] += 0.5
                        W[lj1[j], b2[i]] += 0.5
                        W[lj2[k], b1[i]] += 0.5
                        W[lj1[j], lj2[k]] += 0.5

        W = W + W.transpose()
        return W

    @staticmethod
    def M45_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)

        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and (
                            (A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 0) or (
                            A[lj1[j], b2[i]] == 0 and A[lj2[k], b1[i]] == 1)):
                        W[b1[i], b2[i]] += 1
                        W[lj1[j], b2[i]] += 0.25
                        W[lj2[k], b1[i]] += 0.25

        W = W + W.transpose()
        return W

    @staticmethod
    def M46_edge_degree(A):
        n = len(A)
        W = lil_matrix((n, n), dtype=float)

        b1, b2 = np.where(np.tril(A))

        for i in range(len(b1)):
            lj1 = np.setdiff1d(np.where(A[b1[i], :])[0], b2[i])
            lj2 = np.setdiff1d(np.where(A[b2[i], :])[0], b1[i])

            for j in range(len(lj1)):
                for k in range(len(lj2)):
                    if lj1[j] != lj2[k] and A[lj1[j], lj2[k]] == 1 and A[lj1[j], b2[i]] == 1 and A[lj2[k], b1[i]] == 1:
                        W[b1[i], b2[i]] += 0.5

        W = W + W.transpose()
        return W

