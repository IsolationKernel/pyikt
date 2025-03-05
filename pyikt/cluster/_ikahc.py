"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from typing import Any, Literal, Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyikt.kernel import IsoKernel


class IKAHC(BaseEstimator, ClusterMixin):
    """IKAHC is a novel hierarchical clustering algorithm.
    It uses a data-dependent kernel called Isolation Kernel to measure the similarity between clusters.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int or float or str, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    ik_method: {"inne", "anne"}, default="anne"
        Isolation method to use. The original algorithm in paper is `"anne"`.

    lk_method : {"single", "complete", "average", "weighted"}, default="single"
        The linkage algorithm to use. The supported Linkage Methods are 'single', 'complete', 'average' and
        'weighted'.

    return_flat : bool, default=False
        Whether to return flat clusters that extract from the fitted dendrogram.

    t : float, optional
        The threshold to apply when forming flat clusters.
        Either t or n_clusters should be provided.

    n_clusters : int, optional
        The number of flat clusters to form.
        Either t or n_clusters should be provided.

    criterion : str, default='distance'
        The criterion to use in forming flat clusters. Valid options are
        'distance', 'inconsistent', 'maxclust', or 'monocrit'.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the samples to
        fit the Isolation Kernel.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    isokernel : IsoKernel
        Fitted isolation kernel.

    dendrogram : ndarray
        Cluster hierarchy as computed by scipy.cluster.hierarchy.linkage.

    References
    ----------
    .. [1] Xin Han, Ye Zhu, Kai Ming Ting, and Gang Li,
           "The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms",
           Pattern Recognition, 2023, 139: 109517.

    Examples
    --------
    >>> from pyikt.cluster import IKAHC
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
    >>> clf = IKAHC(n_estimators=200, max_samples=2, lk_method='single', n_clusters=2, return_flat=True)
    >>> clf.fit_predict(X)
    array([1, 2, 1, 1], dtype=int32)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[int, float, str] = "auto",
        lk_method: Literal["single", "complete", "average", "weighted"] = "single",
        ik_method: Literal["inne", "anne"] = "anne",
        return_flat: bool = False,
        t: Optional[float] = None,
        n_clusters: Optional[int] = None,
        criterion: str = "distance",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.ik_method = ik_method
        self.lk_method = lk_method
        self.return_flat = return_flat
        self.t = t
        self.n_clusters = n_clusters
        self.criterion = criterion
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "IKAHC":
        """Fit the IKAHC clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check data
        X = check_array(X, accept_sparse=False)

        # Validate parameters
        if self.lk_method not in ["single", "complete", "average", "weighted"]:
            raise ValueError(
                f"lk_method must be one of 'single', 'complete', 'average', 'weighted', got {self.lk_method}"
            )

        if self.ik_method not in ["inne", "anne"]:
            raise ValueError(
                f"ik_method must be one of 'inne', 'anne', got {self.ik_method}"
            )

        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {self.n_estimators}")

        # Check if both t and n_clusters are provided at initialization
        if self.return_flat and self.t is not None and self.n_clusters is not None:
            raise ValueError(
                "Specify either a distance threshold t or n_clusters, not both."
            )

        # Fit isolation kernel
        self.isokernel_ = IsoKernel(
            method=self.ik_method,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self.isokernel_ = self.isokernel_.fit(X)

        # Calculate similarity matrix and convert to distance matrix (1-similarity)
        similarity_matrix = self.isokernel_.similarity(X)
        self.dendrogram_ = linkage(1 - similarity_matrix, method=self.lk_method)

        if self.return_flat:
            self.labels_ = self._extract_flat_cluster()

        return self

    @property
    def dendrogram(self) -> np.ndarray:
        """Get the dendrogram of the hierarchical clustering.

        Returns
        -------
        dendrogram_ : ndarray
            The dendrogram representing the hierarchical clustering.
        """
        check_is_fitted(self, "dendrogram_")
        return self.dendrogram_

    @property
    def isokernel(self) -> IsoKernel:
        """Get the fitted isolation kernel.

        Returns
        -------
        isokernel_ : IsoKernel
            The fitted isolation kernel.
        """
        check_is_fitted(self, "isokernel_")
        return self.isokernel_

    def _extract_flat_cluster(
        self,
        t: Optional[float] = None,
        n_clusters: Optional[int] = None,
        criterion: Optional[str] = None,
    ) -> np.ndarray:
        """Return cluster labels for each sample based on the hierarchical clustering.

        Parameters
        ----------
        t : float, optional
            The threshold to apply when forming flat clusters.
            Either t or n_clusters should be provided.
        n_clusters : int, optional
            The number of flat clusters to form.
            Either t or n_clusters should be provided.
        criterion : str, optional
            The criterion to use. If not specified, self.criterion is used.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        check_is_fitted(self, "dendrogram_")

        # Use parameters or fall back to instance variables
        t = t if t is not None else self.t
        n_clusters = n_clusters if n_clusters is not None else self.n_clusters
        criterion = criterion if criterion is not None else self.criterion

        if t is not None and n_clusters is not None:
            raise ValueError(
                "Specify either a distance threshold t or n_clusters, not both."
            )

        if t is None and n_clusters is None:
            raise ValueError("Either a threshold t or n_clusters must be provided.")

        if t is not None:
            return fcluster(self.dendrogram_, t=t, criterion=criterion)
        else:
            return fcluster(self.dendrogram_, t=n_clusters, criterion="maxclust")

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram representing the hierarchical clustering.
        """
        self.fit(X)
        return self.dendrogram

    def fit_predict(self, X, y=None):
        """Fit algorithm to data and return the cluster labels."""
        return super().fit_predict(X, y)
