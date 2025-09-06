import numpy as np
import cvxpy as cp

def all_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(i, n):
            if i != j:
                pairs.append((i, j))
    return pairs

def random_weights(n, weighted=True, complete=True):
    nc2 = len(all_pairs(n))
    if weighted:
        w = np.random.rand(nc2)
        if not complete:
            for i in range(len(w)):
                if np.random.randint(0, 2) == 0:
                    w[i] = 0 
    else:
        w = np.random.randint(0, 2, nc2)
    return w

def k_random_pairs(n, k):
    return np.random.choice(range(len(all_pairs(n))), k, replace=False)


