"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, check_random_state

from pyikt.kernel import IsoKernel

from ._kcluster import KCluster


class IDKC(BaseEstimator, ClusterMixin):
    """Isolation Distributional Kernel Clustering.

    A clustering algorithm that leverages Isolation Kernels to transform data into
    a feature space where cluster structures are more distinguishable. The algorithm
    first constructs Isolation Kernel representations, then performs clustering in this
    transformed space using a threshold-based assignment mechanism.

    Parameters
    ----------
    n_estimators : int
        Number of base estimators in the ensemble for the Isolation Kernel.
        Higher values generally lead to more stable results but increase computation time.

    max_samples : int
        Number of samples to draw from X to train each base estimator in the Isolation Kernel.
        Controls the granularity of the kernel representation.

    method : {'inne', 'anne', 'iforest'}
        Method used to calculate the Isolation Kernel:
        - 'inne': Isolation Nearest Neighbor Ensemble
        - 'anne': Approximate Nearest Neighbor Ensemble
        - 'iforest': Isolation Forest

    k : int
        Number of clusters to form in the dataset.

    kn : int
        Number of nearest neighbors used for local contrast density calculation
        during initialization. Higher values consider more neighbors when determining
        density.

    v : float
        Decay factor (0 < v < 1) for reducing the similarity threshold during clustering.
        Smaller values cause faster decay, leading to more aggressive cluster assignments.

    n_init_samples : int
        Number of samples to consider when initializing cluster centers.
        Larger values may produce better initial centers but increase computation.

    init_center : int or array-like of shape (k,), default=None
        Index or indices of initial cluster centers. If None, centers are selected
        automatically based on density and distance considerations.

    is_post_process : bool, default=True
        Whether to perform post-processing refinement of clusters through iterative
        reassignment. Improves cluster quality but adds computational overhead.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm. Pass an int for reproducible results.

    Attributes
    ----------
    clusters_ : list of KCluster objects
        The cluster objects containing assignment and centroid information.

    it_ : int
        Number of iterations performed during the initial clustering phase.

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Points not assigned to any cluster
        have label -1 (outliers).

    is_fitted_ : bool
        Whether the model has been fitted to data.

    Examples
    --------
    >>> from pyikt.cluster import IDKC
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [5, 2], [5, 5],  [1, 0], [5, 0]])
    >>> clustering = IDKC(
    ...     n_estimators=100, max_samples=3, method='anne',
    ...     k=2, kn=5, v=0.5, n_init_samples=4, random_state=42
    ... )
    >>> clustering.fit_predict(X)
    array([1, 1, 0, 0, 1, 0])

    References
    ----------
    .. [1] Ye Zhu, Kai Ming Ting (2023). Kernel-based Clustering via Isolation Distributional Kernel. Information Systems.
    """

    def __init__(
        self,
        n_estimators,
        max_samples,
        method,
        k,
        kn,
        v,
        n_init_samples,
        init_center=None,
        is_post_process=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.method = method
        self.k = k
        self.kn = kn
        self.v = v
        self.n_init_samples = n_init_samples
        self.is_post_process = is_post_process
        self.init_center = init_center
        self.random_state = random_state
        self.clusters_ = []
        self.it_ = 0
        self.labels_ = None
        self.data_index = None

    @property
    def n_it(self):
        """Get number of iterations performed during clustering."""
        return self.it_

    def fit(self, X, y=None):
        """Fit the IDKC clustering model on data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input instances to cluster.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self.data_index = np.arange(X.shape[0])
        isokernel = IsoKernel(
            method=self.method,
            max_samples=self.max_samples,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        data_ik = isokernel.fit_transform(X)
        self._fit(data_ik)

        # Apply post-processing if requested
        if self.is_post_process:
            self._post_process(data_ik)

        self.is_fitted_ = True
        self.labels_ = self._get_labels(X)
        return self

    def _fit(self, X):
        """Perform initial clustering on transformed data."""
        self._initialize_cluster(X)
        # Main clustering loop
        r = None  # Initial threshold
        while self.data_index.size > 0:
            # Get cluster means and calculate similarities
            if sp.issparse(self.clusters_[0].kernel_mean):
                c_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
            else:
                c_mean = np.vstack([c.kernel_mean for c in self.clusters_])

            similarity = safe_sparse_dot(X[self.data_index], c_mean.T)
            tmp_labels = np.argmax(similarity, axis=1).A1
            if sp.issparse(similarity):
                similarity = similarity.todense()
            tmp_similarity = np.max(similarity, axis=1).A1

            # Initialize threshold on first iteration
            if self.it_ == 0:
                r = float(np.max(tmp_similarity))
            r *= self.v
            if np.sum(tmp_similarity) == 0 or r < 0.00001:
                break
            assigned_mask = np.zeros_like(tmp_labels)
            for i in range(self.k):
                I = np.logical_and(tmp_labels == i, tmp_similarity > r)
                if np.sum(I) > 0:
                    self.clusters_[i].add_points(
                        self.data_index[I], X[self.data_index][I]
                    )
                    assigned_mask += I

            self._update_centers(X)
            self.data_index = np.delete(self.data_index, np.where(assigned_mask > 0)[0])
            self.it_ += 1

        return self

    def _initialize_cluster(self, X):
        """Initialize cluster centers based on density and distance criteria."""
        self.clusters_ = [KCluster(i) for i in range(self.k)]
        if self.init_center is None:
            rnd = check_random_state(self.random_state)
            samples_index = rnd.choice(
                self.data_index.size, self.n_init_samples, replace=False
            )
            seeds_id = self._get_seeds(X[samples_index])
            init_center = samples_index[seeds_id]
        else:
            init_center = self.init_center
        for i in range(self.k):
            self.clusters_[i].add_points(init_center[i], X[init_center[i]])
        self.data_index = np.delete(self.data_index, init_center)
        return self

    def _post_process(self, X):
        """Refine clusters through iterative reassignment of points.

        This improves cluster quality by allowing points to move between clusters
        based on similarity until convergence or maximum iterations.
        """
        # Use 1% of data as threshold for stopping
        threshold = max(int(np.ceil(X.shape[0] * 0.01)), 1)

        for _ in range(100):  # Maximum iterations
            old_labels = self._get_labels(X)
            data_index = np.arange(X.shape[0])

            # Get cluster means
            if sp.issparse(self.clusters_[0].kernel_mean):
                c_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
            else:
                c_mean = np.vstack([c.kernel_mean for c in self.clusters_])

            # Compute new assignments
            new_labels = np.argmax(safe_sparse_dot(X, c_mean.T), axis=1)
            if sp.issparse(new_labels):
                new_labels = new_labels.A1

            # Find points that changed clusters
            change_id = new_labels != old_labels

            # Stop if few changes or clusters disappeared
            if np.sum(change_id) < threshold or len(np.unique(new_labels)) < self.k:
                break

            # Update cluster assignments
            old_label, new_label = old_labels[change_id], new_labels[change_id]
            changed_points = data_index[change_id]

            for old_cluster, new_cluster, point_idx in zip(
                old_label, new_label, changed_points
            ):
                self._change_points(
                    self.clusters_[old_cluster],
                    self.clusters_[new_cluster],
                    point_idx,
                    X,
                )

            # Update centers after reassignment
            self._update_centers(X)

        return self

    def _get_seeds(self, X):
        """Select seed points for initialization based on density and distance.

        Uses a strategy similar to Density Peaks clustering to identify points
        that are both high in local density and far from other high-density points.
        """
        dists = 1 - safe_sparse_dot(X, X.T, dense_output=True) / self.n_estimators
        density = self._get_klc(X)
        filter_index = density < density.T
        tmp_dists = np.ones_like(dists)
        tmp_dists[filter_index] = dists[filter_index]
        min_dist = np.min(tmp_dists, axis=1)
        mult = density * min_dist
        sort_mult = np.argpartition(mult, -self.k)[-self.k :]
        return sort_mult

    def _get_klc(self, X):
        """Calculate local contrast density by comparing each point to its nearest neighbors."""
        density = safe_sparse_dot(X, X.mean(axis=0).T)
        n_samples = density.shape[0]
        knn_index = ArgKmin.compute(
            X=X,
            Y=X,
            k=min(self.kn + 1, n_samples),  # Prevent k > n_samples
            metric="sqeuclidean",
            metric_kwargs={},
            strategy="auto",
            return_distance=False,
        )
        nn_density = density[knn_index[:, 1:]].reshape(
            n_samples, min(self.kn, n_samples - 1)
        )
        lc = np.sum(density > nn_density, axis=1)
        return np.array(lc).flatten()

    def _get_labels(self, X):
        """Get cluster labels for all points in the dataset."""
        n_samples = X.shape[0]
        labels = np.full(
            n_samples, -1, dtype=int
        )  # Default to -1 for unassigned points
        for i, cluster in enumerate(self.clusters_):
            labels[cluster.points_] = i
        return labels

    def _change_points(self, source_cluster, target_cluster, point_idx, X):
        """Move a point from one cluster to another, updating both clusters."""
        target_cluster.add_points(point_idx, X[point_idx])
        source_cluster.delete_points(point_idx, X[point_idx])
        return self

    def _update_centers(self, X):
        """Update the center of each cluster to the most central point."""
        for cluster in self.clusters_:
            if len(cluster.points) > 0:
                similarities = safe_sparse_dot(X[cluster.points], cluster.kernel_mean.T)
                center_idx = np.argmax(similarities)
                cluster.set_center(cluster.points[center_idx])
        return self

    def fit_predict(self, X, y=None):
        return super().fit_predict(X, y)
