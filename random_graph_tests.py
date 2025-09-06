import networkx as nx
import completer
import random

def RandomkUnknowns(n, k):
    unknowns = set()
    for _ in range(k):
        i = random.randint(0, n)
        j = random.randint(0, n)
        while (i == j):
            j = random.randint(0, n)
            print(':3')
        while (i, j) or (j, i) in unknowns:
            i = random.randint(0, n)
            j = random.randint(0, n)
            print(':33')
            while (i == j):
                j = random.randint(0, n)
                print(':333')
        if i > j:
            temp = i
            i = j
            j = temp
        unknowns.add((i, j))

    return unknowns

def TestCompleter(G, k):

    # Random choice of unknown weights
    unknowns = RandomkUnknowns(len(G), k)

    L = G.laplacian_matrix()
    real_weights = []
    for i, j in unknowns:
        real_weights.append(L[i][j])

    est_weights = completer.CompleteGraph(G, unknowns)

    print(real_weights)
    print(est_weights)


