import cvxpy as cp
import numpy as np
import networkx as nx
import laplacian as Lap

def CompleteGraph(G, unknowns):
    # Input:
    #   G <- NetworkX Graph of n nodes
    #   unknowns <- Set of k piars of unknown indices
    # Output:
    #   w <- Vector of k weights on unknown indices

    n = len(G)          # Number of vertices
    k = len(unknowns)   # Number of unknown indices

    r = []
    # Get vector of resistances on unknowns
    for (i, j) in unknowns:
        r.append(nx.resistance_distance(G, i, j))

    # Laplacian of G
    L_true = nx.linalg.laplacian_matrix(G).toarray()

    # Laplacian of G with unknown pairs set to 0
    #   or equivalently "partial information Laplacian"
    L_incomplete = L_true
    for i, j in unknowns:
        L_incomplete[i][j] = 0
        L_incomplete[j][i] = 0

    # Regularize incomplete laplacian
    L_incomplete = Lap.MakeInvertible(L_incomplete)

    # Vector of k-unknown weights
    w_est = cp.Variable(k)

    # Estimated completed Laplacian
    L_est = L_incomplete
    for l, (i, j) in enumerate(unknowns):
        L_est = L_est + (w_est[l] * Lap.EdgeLaplacian(n, i, j))

    # Setting up optim problem
    f = (r @ w_est) - cp.log_det(L_est)
    objective = cp.Minimize(f)
    constraints = [w_est >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Regularize solution
    output = []
    for v in w_est.value:
        output.append(v)
    return output
