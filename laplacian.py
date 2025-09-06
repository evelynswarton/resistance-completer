import numpy as np

# Get the inveritble Laplacian based on L
def MakeInvertible(L):
    return L + ((1 / len(L)) * np.ones((len(L), len(L))))

# Get the original Laplacian based on invertible Laplacian
def ReverseInvertible(tilde_L):
    return tilde_L - ((1 / len(L)) * np.ones((len(L), len(L))))

# Spectrum of graph Laplacian
def Spectrum(L):
    return np.linalg.eig(L)[0]

def EdgeLaplacian(n, i, j):
    Lij = np.zeros((n, n))
    Lij[i][j] = -1
    Lij[j][i] = -1
    Lij[i][i] = 1
    Lij[j][j] = 1
    return Lij


