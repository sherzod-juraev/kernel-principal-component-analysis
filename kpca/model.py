from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from numpy import ndarray, exp, full, sqrt


class KernelPCA:
    """
    RBF kernel PCA implementation

    Parameters
    ------------

    X: {NumPy ndarray},
        shape = [n_examples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    ------------
    X_pc: {NumPy ndarray},
        shape = [n_examples, k_features]
        Projected dataset
    """

    def __init__(self, *, gamma: float, n_components: int):

        self.gamma = gamma
        self.n_components = n_components

    def fit(self, X: ndarray, /) -> ndarray:

        sq_dists = pdist(X, metric='sqeuclidean')
        matrix_sq_dists = squareform(sq_dists)
        K = exp(-self.gamma * matrix_sq_dists)
        one_N_matrix = full((X.shape[0], X.shape[0]), 1 / X.shape[0])
        K_c = K - one_N_matrix.dot(K) - K.dot(one_N_matrix) + one_N_matrix.dot(K).dot(one_N_matrix)
        eigenvals, eigenvecs = eigh(K_c)
        eigenvals, eigenvecs = eigenvals[::-1], eigenvecs[:, ::-1]
        X_projected = eigenvecs[:, :self.n_components] / sqrt(eigenvals[:self.n_components])
        return X_projected