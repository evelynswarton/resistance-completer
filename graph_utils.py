import numpy as np
import cvxpy as cp

def edge_laplacian(n, i, j):
    A = np.zeros((n, n))
    A[i][i] = 1
    A[j][j] = 1
    A[i][j] = -1
    A[j][i] = -1
    return A

def ordered_index_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(i, n):
            if i != j:
                pairs.append((i, j))
    return pairs

def random_weights(n, weighted=True, complete=True):
    nc2 = len(ordered_index_pairs(n))
    if weighted:
        w = np.random.rand(nc2)
        if not complete:
            for i in range(len(w)):
                if np.random.randint(0, 2) == 0:
                    w[i] = 0 
    else:
        w = np.random.randint(0, 2, nc2)
    return w

def num_pairs(n):
    return int(n * (n - 1) // 2)

def erdos_renyi_Gnp(n, p):
    m = num_pairs(n)
    w = np.random.rand(m)
    for i, wi in enumerate(w):
        if wi <= p:
            w[i] = 1
        else:
            w[i] = 0
    return w

def erdos_renyi_Gnm(n, m):
    sampled = np.random.choice(num_pairs(n), m, replace=False)
    w = np.zeros(num_pairs(n))
    for edge in sampled:
        w[edge] = 1
    return w

def regularize(L):
    n = len(L)
    return L + ((1 / n) * np.ones((n, n)))

def array_mask(arr, mask):
    result = []
    for index in mask:
        result.append(arr[index])
    return result
