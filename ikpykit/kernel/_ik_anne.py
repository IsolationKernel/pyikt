"""
ikpykit (c) by Xin Han

ikpykit is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max


class IK_ANNE(TransformerMixin, BaseEstimator):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses Voronoi diagrams to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

    Parameters
    ----------

    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : int, default=256
        The number of samples to draw from X to train each base estimator.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    Attributes
    ----------
    center_data : ndarray of shape (n_unique_centers, n_features)
        The unique center points data used for constructing Voronoi cells.

    unique_ids : ndarray
        Indices of unique center points in the original dataset.

    center_ids : ndarray of shape (n_estimators, max_samples_)
        Indices of center points selected for each estimator.

    is_fitted_ : bool
        Whether the estimator has been fitted.

    References
    ----------
    .. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C.
    "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering".
    In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762
    """

    def __init__(self, n_estimators=100, max_samples=256, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        n_samples = X.shape[0]
        self.max_samples_ = min(self.max_samples, n_samples)
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        # Select center points for each estimator
        self.center_ids = np.empty(
            (self.n_estimators, self.max_samples_), dtype=np.int32
        )
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            self.center_ids[i] = rnd.choice(n_samples, self.max_samples_, replace=False)

        # Only store unique center points to save memory
        self.unique_ids = np.unique(self.center_ids)
        self.center_data = X[self.unique_ids]

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Compute the isolation kernel feature of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        sparse matrix: The finite binary features based on the kernel feature map.
            The features are organized as a n_instances by (n_estimators * max_samples_) matrix.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = X.shape[0]
        n_features = self.n_estimators * self.max_samples_

        # Precompute all distances to center points once
        X_dists = euclidean_distances(X, self.center_data)

        # Create lookup dictionary for distances
        id_to_index = {id_val: idx for idx, id_val in enumerate(self.unique_ids)}

        # Prepare for sparse matrix construction
        rows = np.tile(np.arange(n_samples), self.n_estimators)
        cols = np.zeros(n_samples * self.n_estimators, dtype=np.int32)
        data = np.ones(n_samples * self.n_estimators, dtype=np.float64)

        # Process each estimator
        for est_idx in range(self.n_estimators):
            centers = self.center_ids[est_idx]
            # Map center IDs to positions in unique_ids array
            center_indices = np.array([id_to_index[center] for center in centers])

            # Get distances to centers for this estimator
            estimator_dists = X_dists[:, center_indices]

            # Find nearest center for each sample
            nn_indices = np.argmin(estimator_dists, axis=1)

            # Calculate column indices
            start_idx = est_idx * n_samples
            end_idx = (est_idx + 1) * n_samples
            cols[start_idx:end_idx] = nn_indices + (est_idx * self.max_samples_)

        # Create sparse matrix
        embedding = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_samples, n_features)
        )

        return embedding
