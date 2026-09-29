import cvxpy as cp
import json
import core.graph_utils as Graph
import core.laplace_utils as Laplacian
import core.completion as Completion
import numpy as np
import networkx as nx
import random

SEED = 0
rng = np.random.default_rng(SEED)

def test_completion_gnp(n, p, k, seed):
    L = nx.linalg.laplacian_matrix(nx.gnp_random_graph(n, p, seed=seed)).toarray()
    if Laplacian.IsConnected(L):
        unknowns = Graph.k_random_pairs(len(L), k)
        solution = Completion.resistance_completion(L, unknowns)
        error = np.linalg.norm(Laplacian.ArrayMask(Laplacian.Weights(L), unknowns) - solution)
        return error, True
    else:
        return -1, False

def run_gnp_tests():
    test_results = dict()
    for n in range(3, 10):
        test_results[n] = dict()
        for p in np.arange(0, 1, 0.05):
            test_results[n][p] = dict()
            for k in range(1, len(Graph.all_pairs(n))):
                test_results[n][p][k] = test_completion_gnp(n, p, k, rng)
                print(f'n=[{n}], p=[{p}], k=[{k}]')
                print(test_results[n][p][k])

    with open('random_graph_tests.json', 'w') as f:
        json.dump(test_results, f)
