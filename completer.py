import cvxpy as cp
import numpy as np

def CompleteGraph(G, unknowns):
    # G <- NetworkX graph
    # unknowns <- set of sets of indices of size 2

    # Number of nodes
    n = len(G)

    # Number of unknowns
    k = len(unknowns)

    # Get vector of resistances on unknowns
    r = []
    for i, j in unknowns:
        r.append(G.resistance_distance(i, j))

    # Graph laplacian
    L_true = nx.linalg.laplacematrix(G)

    # Get incomplete version of Laplacian
    L_incomplete = L_true
    for i, j in unknowns:
        L_incomplete[i][j] = 0
        L_incomplete[j][i] = 0

    # Regularize incomplete laplacian
    L_incomplete = regularize(L_incomplete)

    # Vector of k-unknown weights
    w_est = cp.Variable(k)

    # Estimated completed Laplacian
    L_est = L_incomplete
    for l in range(k):
        i, j = unknowns[l]
        L_est += w_est[l] * edge_laplacian(n, i, j)

    # Setting up optim problem
    f = r @ w_est - cp.log_det(L_est)
    objective = cp.Minimize(f)
    constraints = [w_est >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Regularize solution
    output = []
    for v in w_est.value
        output.append(v - (1/n))
    return output


def solve():
    pairs = ordered_index_pairs(n)
    L = np.zeros((n, n))
    for idx, (i, j) in enumerate(pairs):
        L = L + (w[idx] * edge_laplacian(n, i, j))
    R = np.zeros((n,n))
    Lpls = np.linalg.pinv(L)
    for idx, (i, j) in enumerate(pairs):
        R[i][j] = Lpls[i][i] + Lpls[j][j] - (2 * Lpls[i][j])
        R[j][i] = R[i][j]
    unknowns = np.random.choice(range(len(pairs)), k, replace=False)
    knowns = []
    for i in range(len(pairs)):
        if i not in unknowns:
            knowns.append(i)
    unknown_pairs = array_mask(pairs, unknowns)
    known_pairs = array_mask(pairs, knowns)
    weights_known = array_mask(w, knowns)
    weights_unknown = cp.Variable(k, nonneg=True)
    L_known = np.zeros((n, n))
    for idx, wi in enumerate(weights_known):
        i, j = known_pairs[idx]
        L_known = L_known + (wi * edge_laplacian(n, i, j)) 
    L_known = regularize(L_known)
    L_completed = L_known
    for idx, wi in enumerate(weights_unknown):
        i, j = unknown_pairs[idx]
        L_completed = L_completed + (wi * edge_laplacian(n, i, j)) 
    rhs = cp.log_det(L_completed)

    r = np.zeros(k)
    for idx, (i, j) in enumerate(unknown_pairs):
        r[idx] = R[i][j]
    lhs = r @ weights_unknown

    objective = cp.Minimize(lhs - rhs)
    constraints = [weights_unknown >= 0]
    problem = cp.Problem(objective, constraints)

    #print(f'solving for [{k}] weights on [{n}] x [{n}] graph...')
    problem.solve()
    error_vec = weights_unknown.value - array_mask(w, unknowns)
    #print(f'  ERROR_TERM : {error_vec}')
    #print(f'NORMED_ERROR : {np.linalg.norm(error_vec)}')
    return np.linalg.norm(error_vec)

