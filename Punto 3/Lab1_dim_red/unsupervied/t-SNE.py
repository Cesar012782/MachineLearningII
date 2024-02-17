import numpy as np
from sklearn.manifold import TSNE

class TSNE:
    def __init__(self, n_components, perplexity=30.0, learning_rate=200.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.embedding = None

    def fit_transform(self, X):
        model = TSNE(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate)
        self.embedding = model.fit_transform(X)
        return self.embedding

