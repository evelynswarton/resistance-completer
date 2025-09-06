import cvxpy as cp
import numpy as np
import networkx as nx
import random
import laplace_utils as Laplacian
import graph_utils as Graph

def array_mask(arr, mask):
    result = []
    for index in mask:
        result.append(arr[index])
    return result

def resistance_completion(L, unknowns):
    n = len(L)
    k = len(unknowns)
    pairs = Graph.all_pairs(n)
    w = Laplacian.Weights(L)
    R = Laplacian.ResistanceMatrix(L)
    knowns = []
    for i in range(len(pairs)):
        if i not in unknowns:
            knowns.append(i)
    # Vector of resistances on unknowns
    unknown_pairs = array_mask(pairs, unknowns)
    known_pairs = array_mask(pairs, knowns)

    weights_known = array_mask(w, knowns)
    weights_unknown = cp.Variable(k, nonneg=True)

    L_known = np.zeros((n, n))
    for idx, wi in enumerate(weights_known):
        i, j = known_pairs[idx]
        L_known = L_known + (wi * Laplacian.Edge(n, i, j)) 
    L_known = Laplacian.Invertible(L_known)
    L_completed = L_known
    for idx, wi in enumerate(weights_unknown):
        i, j = unknown_pairs[idx]
        L_completed = L_completed + (wi * Laplacian.Edge(n, i, j)) 
    rhs = cp.log_det(L_completed)

    r = np.zeros(k)
    for idx, (i, j) in enumerate(unknown_pairs):
        r[idx] = R[i][j]
    lhs = r @ weights_unknown

    objective = cp.Minimize(lhs - rhs)
    constraints = [weights_unknown >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    formatted_results = [f'{v:.5f}' for v in weights_unknown.value]
    # print(f'weights obtained: {formatted_results}\n    true weights: {array_mask(w, unknowns)}')
    return weights_unknown.value

def test_completion_gnp(n, p, k):
    L = nx.linalg.laplacian_matrix(nx.gnp_random_graph(n, p)).toarray()
    unknowns = Graph.k_random_pairs(len(L), k)
    solution = resistance_completion(L, unknowns)
    error = np.linalg.norm(array_mask(Laplacian.Weights(L), unknowns) - solution)
    is_connected = Laplacian.IsConnected(L)
    return error, is_connected

for i in range(10):
    print(test_completion_gnp(20, i / 10, 80))
