import numpy as np
from sklearn.manifold import TSNE

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 max_iter=1000, early_exaggeration=4.0, early_exaggeration_iter=250,
                 exaggeration_decay=0.95, momentum=0.8, random_state=None, verbose=0):
        """
        Parameters
        ----------
        n_components : int, optional (default: 2)
            Dimension of the embedded space.
        perplexity : float, optional (default: 30)
            The perplexity is related to the number of nearest neighbors that
            is used in other manifold learning algorithms. Larger datasets
            usually require a larger perplexity. Consider selecting a value
            between 5 and 50. The choice is not extremely critical since t-SNE
            is quite insensitive to this parameter.
        learning_rate : float, optional (default: 200.0)
            The learning rate for t-SNE is usually in the range [10.0, 1000.0].
            If the learning rate is too high, the data may look like a ‘ball’
            with any point approximately equidistant to its nearest neighbours.
            If the learning rate is too low, most points may look compressed
            in a dense cloud with few outliers.
        max_iter : int, optional (default: 1000)
            Maximum number of iterations for the optimization. Should be at
            least 250.
        early_exaggeration : float, optional (default: 4.0)
            Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them. For
            larger values, the space between natural clusters will be larger
            in the embedded space. Again, the choice of this parameter is not
            very critical. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high.
        early_exaggeration_iter : int, optional (default: 250)
            Number of iterations for which early exaggeration is applied.
        exaggeration_decay : float, optional (default: 0.95)
            Exaggeration factor for each iteration after early exaggeration.
        momentum : float, optional (default: 0.8)
            Momentum for gradient descent update.
        random_state : int, RandomState instance or None, optional (default: None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose : int, optional (default: 0)
            Verbosity level.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.exaggeration_decay = exaggeration_decay
        self.momentum = momentum
        self.random_state = random_state
        self.verbose = verbose
        self.embedding = None

    def fit_transform(self, X):
        """
        Fit X into an embedded space and return that transformed output.
        """
        model = TSNE(**self.__dict__)
        self.embedding = model.fit_transform(X)
        return self.embedding


