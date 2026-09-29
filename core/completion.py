import cvxpy as cp
import core.graph_utils as Graph
import core.laplace_utils as Laplacian
import numpy as np
import networkx as nx

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
    unknown_pairs = Laplacian.ArrayMask(pairs, unknowns)
    known_pairs = Laplacian.ArrayMask(pairs, knowns)

    weights_known = Laplacian.ArrayMask(w, knowns)
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
    return weights_unknown.value

