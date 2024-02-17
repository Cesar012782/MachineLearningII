import numpy as np
from scipy.linalg import svd

class PCA:
    def __init__(self, n_components, solver="svd"):
        self.n_components = n_components
        self.solver = solver
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        _, _, self.components = svd(X_centered, full_matrices=True)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components[:self.n_components].T)

    def explained_variance_ratio(self):
        s_squared = np.square(self.components)
        variance_ratio = s_squared / np.sum(s_squared, axis=0)
        return variance_ratio[:self.n_components]
