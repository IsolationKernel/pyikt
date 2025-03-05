"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from typing import Any, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from pyikt.cluster import IDKC
from pyikt.group.utils import check_format
from pyikt.kernel import IsoKernel


class TIDKC(BaseEstimator, ClusterMixin):
    """Trajectory Isolation Distributional Kernel Clustering (TIDKC).

    TIDKC identifies non-linearly separable clusters with irregular shapes and varied
    densities in trajectory data using distributional kernels. It operates in linear
    time, does not rely on random initialization, and is robust to outliers.

    Parameters
    ----------
    k : int
        The number of clusters to form.

    kn : int
        The number of nearest neighbors to consider when calculating the local contrast.

    v : float
        The decay factor for reducing the threshold value.

    n_init_samples : int
        The number of samples to use for initializing the cluster centers.

    n_estimators_1 : int, default=100
        Number of base estimators in the first step ensemble.

    max_samples_1 : int, float or "auto", default="auto"
        Number of samples to draw for training each base estimator in first step:
        - If int, draws exactly `max_samples_1` samples
        - If float, draws `max_samples_1 * n_samples` samples
        - If "auto", draws `min(8, n_samples)` samples

    n_estimators_2 : int, default=100
        Number of base estimators in the second step ensemble.

    max_samples_2 : int, float or "auto", default="auto"
        Number of samples to draw for training each base estimator in second step:
        - If int, draws exactly `max_samples_2` samples
        - If float, draws `max_samples_2 * n_samples` samples
        - If "auto", draws `min(8, n_samples)` samples

    method : {"inne", "anne"}, default="anne"
        Isolation method to use. "anne" is the original algorithm from the paper.

    is_post_process : bool, default=True
        Whether to perform post-processing to refine the clusters.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset.

    iso_kernel_ : IsoKernel
        The fitted isolation kernel.

    idkc_ : IDKC
        The fitted IDKC clustering model.

    References
    ----------
    .. [1] Z. J. Wang, Y. Zhu and K. M. Ting, "Distribution-Based Trajectory Clustering,"
           2023 IEEE International Conference on Data Mining (ICDM).

    Examples
    --------
    >>> from pyikt.trajectory import TIDKC
    >>> from pyikt.trajectory.dataloader import SheepDogs
    >>> sheepdogs = SheepDogs()
    >>> X, y = sheepdogs.load(return_X_y=True)
    >>> clf = TIDKC(k=2, kn=5, v=0.5, n_init_samples=10).fit(X)
    >>> predictions = clf.fit_predict(X)
    """

    def __init__(
        self,
        k: int,
        kn: int,
        v: float,
        n_init_samples: int,
        n_estimators_1: int = 100,
        max_samples_1: Union[int, float, str] = "auto",
        n_estimators_2: int = 100,
        max_samples_2: Union[int, float, str] = "auto",
        method: Literal["inne", "anne"] = "anne",
        is_post_process: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_estimators_1 = n_estimators_1
        self.max_samples_1 = max_samples_1
        self.n_estimators_2 = n_estimators_2
        self.max_samples_2 = max_samples_2
        self.method = method
        self.k = k
        self.kn = kn
        self.v = v
        self.n_init_samples = n_init_samples
        self.is_post_process = is_post_process
        self.random_state = random_state

    def fit(self, X: list, y: Any = None) -> "TIDKC":
        """Fit the trajectory cluster model.

        Parameters
        ----------
        X : array-like of shape (n_trajectories, n_points, n_features)
            The input trajectories to train on.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If method is not valid.
        """
        X = check_format(X, n_features=2)

        # Validate method parameter
        if self.method not in ["inne", "anne"]:
            raise ValueError(
                f"method must be one of 'inne', 'anne', got: {self.method}"
            )

        # Fit the model
        self._fit(X)
        self.is_fitted_ = True
        return self

    def _kernel_mean_embedding(self, X, t):
        """Calculate kernel mean embedding of transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_estimators)
            Transformed data from isolation kernel
        t : int
            Normalization factor (typically number of estimators)

        Returns
        -------
        embedding : ndarray
            Kernel mean embedding
        """
        return np.mean(X, axis=0) / t

    def _fit(self, X):
        """Internal fitting method.

        Parameters
        ----------
        X : array-like of shape (n_trajectories, n_points, n_features)
            The input trajectories to train on.
        """
        X_full = np.vstack(X)
        iso_kernel = IsoKernel(
            n_estimators=self.n_estimators_1,
            max_samples=self.max_samples_1,
            method=self.method,
            random_state=self.random_state,
        )
        iso_kernel.fit(X_full)
        split_idx = np.cumsum([len(x) for x in X])
        X_trans = iso_kernel.transform(X_full)
        group_embeddings = np.split(X_trans.toarray(), split_idx[:-1], axis=0)
        X_embeddings = np.asarray(
            [
                self._kernel_mean_embedding(x, self.n_estimators_1)
                for x in group_embeddings
            ]
        )
        self.idkc_ = IDKC(
            n_estimators=self.n_estimators_2,
            max_samples=self.max_samples_2,
            method=self.method,
            k=self.k,
            kn=self.kn,
            v=self.v,
            n_init_samples=self.n_init_samples,
            is_post_process=self.is_post_process,
            random_state=self.random_state,
        )

        self.labels_ = self.idkc_.fit_predict(X_embeddings)
        return self

    def fit_predict(self, X, y=None):
        """Fit the model and predict clusters for X.

        Parameters
        ----------
        X : array-like of shape (n_trajectories, n_points, n_features)
            The input trajectories.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
