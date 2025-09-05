from graph_utils import *


def solve(n, w, k):
    #print('setting up graph...')
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


num_verts = 3
num_probabilities = 10
num_tests = 5

for k in range(1, num_verts):
    for p in range(num_probabilities):
        p = p / num_probabilities
        errs = []
        for i in range(num_tests):
            w = erdos_renyi_Gnp(num_verts, p)
            errs.append(solve(num_verts, w, k))
        with open(f'n_{num_verts}__k_{k}__p_{p:2.2f}.log', 'w') as f:
            f.write(','.join([str(err) for err in errs]))

