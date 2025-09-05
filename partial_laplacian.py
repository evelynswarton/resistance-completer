class PartialGraph:
    def __init__(self, laplacian, known_indices, weighted=False):
        self.num_vertices = len(laplacian)
        self.laplacian = laplacian
        self.known_indices = known_indices

    def degree_mat(self):
        return np.diag(np.diag(self.laplacian))

    def adjacency_mat(self):
        return self.laplacian - self.degree_mat()

    def all_pairs(self):
        all_pairs = set()
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                ij_pair = {i, j}
                all_pairs.add(ij_pair)
        return all_pairs

    def known_pairs(self):
        known_pairs = set()
        for pair in self.known_indices:
            i = pair[0]
            j = pair[1]
            ij_pair = {i, j}
            known_pairs.add(ij_pair)
        return known_pairs

    def unknown_pairs(self):
        return self.all_pairs().difference(self.known_pairs())


