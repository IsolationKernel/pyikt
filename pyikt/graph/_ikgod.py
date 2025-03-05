"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import copy
import numbers
from warnings import warn

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from pyikt.graph.utils import check_format, get_degrees, get_neighbors
from pyikt.kernel import IsoKernel


class IKGOD(BaseEstimator):
    """Isolation-based Graph Anomaly Detection using kernel embeddings.

    This algorithm detects anomalies in graphs by using isolation kernels on subgraph
    features. It combines graph structure and node features to identify outliers.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of isolation estimators in the ensemble.

    max_samples : int, float or "auto", default="auto"
        Number of samples to draw for training each base estimator:
        - If int, draw `max_samples` samples
        - If float, draw `max_samples * X.shape[0]` samples
        - If "auto", use `min(16, n_samples)`

    contamination : float or "auto", default="auto"
        Expected proportion of outliers in the data:
        - If "auto", threshold is set at -0.5 as in the original paper
        - If float, must be in range (0, 0.5]

    method : {"inne", "anne", "auto"}, default="inne"
        Isolation method to use. The original algorithm uses "inne".

    random_state : int, RandomState or None, default=None
        Controls randomness for reproducibility.

    h : int, default=3
        Maximum hop distance for subgraph extraction.

    Attributes
    ----------
    max_samples_ : int
        Actual number of samples used

    embedding_ : array of shape (n_samples, n_features)
        Learned subgraph embeddings

    offset_ : float
        Threshold for determining outliers

    is_fitted_ : bool
        Whether the model has been fitted

    References
    ----------
    .. [1] Zhong Zhuang, Kai Ming Ting, Guansong Pang, Shuaibin Song (2023).
       Subgraph Centralization: A Necessary Step for Graph Anomaly Detection.
       Proceedings of The SIAM Conference on Data Mining.

    Examples
    --------
    >>> from pyikt.graph import IKGOD
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> # Create adjacency matrix and features
    >>> adj = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> features = np.array([[0.1, 0.2], [0.3, 0.4], [5.0, 6.0]])
    >>> # Fit model
    >>> model = IKGOD(n_estimators=100, h=2).fit(adj, features)
    >>> # Predict outliers
    >>> lables = model.predict(features)
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        method="inne",
        random_state=None,
        h=3,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.method = method
        self.h = h

    def fit(self, adjacency, features, y=None):
        """Fit the IKGOD model.

        Parameters
        ----------
        adjacency : array-like or sparse matrix of shape (n_samples, n_samples)
            Adjacency matrix of the graph

        features : array-like of shape (n_samples, n_features)
            Node features

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check and format inputs
        adjacency = check_format(adjacency)
        features = check_array(features, accept_sparse=False)

        n_samples = features.shape[0]

        # Determine max_samples
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(16, n_samples)
            else:
                raise ValueError(
                    f"max_samples '{self.max_samples}' is not supported. "
                    f'Valid choices are: "auto", int or float'
                )
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    f"max_samples ({self.max_samples}) is greater than the "
                    f"total number of samples ({n_samples}). max_samples "
                    f"will be set to n_samples for estimation."
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    f"max_samples must be in (0, 1], got {self.max_samples}"
                )
            max_samples = int(self.max_samples * n_samples)

        self.max_samples_ = max_samples

        # Fit the model
        self._fit(adjacency, features)
        self.is_fitted_ = True

        # Set contamination threshold
        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    f"contamination must be in (0, 0.5], got: {self.contamination}"
                )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper
            self.offset_ = -0.5
        else:
            # Set threshold based on contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(features), 100.0 * self.contamination
            )

        return self

    def _fit(self, adjacency, features):
        """Internal fitting method.

        Parameters
        ----------
        adjacency : scipy.sparse matrix
            Adjacency matrix of the graph

        features : array-like
            Node features

        """
        # Transform features using isolation kernel
        iso_kernel = IsoKernel(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples_,
            random_state=self.random_state,
            method=self.method,
        )
        iso_kernel = iso_kernel.fit(features)
        features_trans = iso_kernel.transform(features, dense_output=True)

        # Extract h-hop subgraphs for each node
        h_index = self._get_h_nodes_n_dict(adjacency)

        # Compute subgraph embeddings
        self.embedding_ = self._subgraph_embeddings(adjacency, features_trans, h_index)
        self.is_fitted_ = True
        return self

    def _get_h_nodes_n_dict(self, adj):
        """Extract h-hop neighbors for each node.

        Parameters
        ----------
        adj : scipy.sparse matrix
            Adjacency matrix of the graph

        Returns
        -------
        h_index : list of lists
            h_index[i] contains node indices in the h-hop neighborhood of node i
        """
        adj_h = sp.eye(adj.shape[0])
        M = [{i: 0} for i in range(adj.shape[0])]
        h_index = [[i] for i in range(adj.shape[0])]

        # Iteratively expand neighborhoods
        for k in range(self.h):
            adj_h = sp.coo_matrix(adj_h * adj)
            for i, j in zip(adj_h.row, adj_h.col):
                if j not in M[i]:
                    M[i][j] = k + 1
                    h_index[i].append(j)
        return h_index

    def _subgraph_embeddings(self, adjacency, features, subgraph_index):
        """Compute embeddings for each node's subgraph.

        Parameters
        ----------
        adjacency : scipy.sparse matrix
            Adjacency matrix of the graph

        features : array-like
            Node features

        subgraph_index : list of lists
            Subgraph node indices

        Returns
        -------
        subgraph_embedding : scipy.sparse matrix
            Matrix of subgraph embeddings
        """
        n_nodes = adjacency.shape[0]
        subgraph_embedding = None

        for i in range(n_nodes):
            source_feat = features[i, :]
            subgraph_feat = features[subgraph_index[i]]

            # Center features around source node
            subgraph_feat = subgraph_feat - np.tile(
                source_feat, (len(subgraph_index[i]), 1)
            )
            adj_i = adjacency[subgraph_index[i], :][:, subgraph_index[i]]
            graph_embed = self._wlembedding(adj_i, subgraph_feat)
            if subgraph_embedding is None:
                subgraph_embedding = graph_embed
            else:
                subgraph_embedding = sp.vstack((subgraph_embedding, graph_embed))
        return subgraph_embedding

    def _wlembedding(self, adjacency, X):
        """Compute Weisfeiler-Lehman embedding for a subgraph.

        Parameters
        ----------
        adjacency : scipy.sparse matrix
            Adjacency matrix of the subgraph

        X : array-like
            Node features

        Returns
        -------
        embedding : scipy.sparse matrix
            WL embedding
        """
        n_nodes = adjacency.shape[0]
        degrees = get_degrees(adjacency)
        tmp_embedding = X
        embedding = copy.deepcopy(X)

        # Iterate WL algorithm
        for _ in range(1, self.h + 1):
            updated_embedding = np.empty(X.shape)
            for i in range(n_nodes):
                neighbors = get_neighbors(adjacency, i)
                if degrees[i] > 0:  # Avoid division by zero
                    updated_embedding[i] = (
                        tmp_embedding[neighbors].sum(axis=0) / degrees[i]
                        + tmp_embedding[i]
                    ) / 2
                else:
                    updated_embedding[i] = tmp_embedding[i]

            tmp_embedding = check_format(updated_embedding)
            embedding = sp.hstack((embedding, tmp_embedding))

        # Return mean embedding
        embedding = check_format(embedding.mean(axis=0))
        return embedding

    def _kernel_mean_embedding(self, X):
        """Compute kernel mean embedding.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        kme : array
            Kernel mean embedding
        """
        return np.mean(X, axis=0) / self.max_samples_

    def predict(self, X):
        """Predict outliers in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            +1 for inliers, -1 for outliers
        """
        check_is_fitted(self, "is_fitted_")
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """Compute decision function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision scores. Negative scores represent outliers.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """Compute anomaly scores for samples.

        Lower scores indicate more anomalous points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Lower values indicate more anomalous points.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=False)
        kme = self._kernel_mean_embedding(self.embedding_)
        scores = safe_sparse_dot(self.embedding_, kme.T).A1
        return -scores
