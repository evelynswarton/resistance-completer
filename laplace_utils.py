import graph_utils as Graph
import numpy as np
import networkx as nx

def Edge(n, i, j):
    Lij = np.zeros((n, n))
    Lij[i][j] = -1
    Lij[j][i] = -1
    Lij[i][i] = 1
    Lij[j][j] = 1
    return Lij

def Invertible(L):
    return L + ((1 / len(L)) * np.ones((len(L), len(L))))

def Weights(L):
    pairs = Graph.all_pairs(len(L))
    w = []
    for i, j in pairs:
        w.append(-L[i][j])
    return w

def ResistanceMatrix(L):
    R = np.zeros((len(L), len(L)))
    L_pinv = np.linalg.pinv(L)
    for idx, (i, j) in enumerate(Graph.all_pairs(len(L))):
        R[i][j] = L_pinv[i][i] + L_pinv[j][j] - (2 * L_pinv[i][j])
        R[j][i] = R[i][j]
    return R

def IsConnected(L):
    D = np.diag(np.diag(L))
    A = D - L
    G = nx.Graph(A)
    return nx.is_connected(G)
