"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from pyikt.group.utils import check_format
from pyikt.kernel import IsoKernel


class IKGAD(OutlierMixin, BaseEstimator):
    """Isolation Kernel-based Group Anomaly Detection.

    IKGAD applies isolation kernel techniques to detect anomalies in groups of data points.
    It leverages a two-step approach: first transforming the data using an isolation kernel,
    then calculating kernel mean embeddings for each group to detect anomalous groups.
    The algorithm is effective for detecting both global and local group anomalies.

    Parameters
    ----------
    n_estimators_1 : int, default=200
        The number of base estimators in the first-level ensemble.

    max_samples_1 : int, float, or "auto", default="auto"
        The number of samples to draw for training each first-level base estimator:

        - If int, draws exactly `max_samples_1` samples
        - If float, draws `max_samples_1 * X.shape[0]` samples
        - If "auto", uses `min(8, n_samples)`

    n_estimators_2 : int, default=200
        The number of base estimators in the second-level ensemble.

    max_samples_2 : int, float, or "auto", default="auto"
        The number of samples to draw for training each second-level base estimator:

        - If int, draws exactly `max_samples_2` samples
        - If float, draws `max_samples_2 * X.shape[0]` samples
        - If "auto", uses `min(8, n_samples)`

    method : {"inne", "anne", "auto"}, default="inne"
        Isolation method to use. The "inne" option corresponds to the approach
        described in the original paper.

    contamination : "auto" or float, default="auto"
        Proportion of outliers in the data set:

        - If "auto", the threshold is determined as in the original paper
        - If float, should be in range (0, 0.5]

    random_state : int, RandomState instance or None, default=None
        Controls the random seed for reproducibility.

    Attributes
    ----------
    iso_kernel_1_ : IsoKernel
        First-level trained isolation kernel.

    offset_ : float
        Decision threshold for outlier detection.

    References
    ----------
    .. [1] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2022).
       Isolation Distributional Kernel: A new tool for kernel based point and group anomaly detections.
       IEEE Transactions on Knowledge and Data Engineering.

    Examples
    --------
    >>> from pyikt.group import IKGAD
    >>> import numpy as np
    >>> X =[[[1.0, 1.1], [1.2, 1.3]], [[1.3, 1.2], [1.1, 1.0]], [[1.0, 1.2], [1.4, 1.3]], [[5.0, 5.1], [5.2, 5.3]]]
    >>> clf = IKGAD(max_samples_1=2, max_samples_2=2, contamination=0.25, random_state=42)
    >>> clf = clf.fit(X)
    >>> clf.predict(X)
    array([ 1,  1,  1, -1])
    """

    def __init__(
        self,
        n_estimators_1=200,
        max_samples_1="auto",
        n_estimators_2=200,
        max_samples_2="auto",
        method="inne",
        contamination="auto",
        random_state=None,
    ):
        self.n_estimators_1 = n_estimators_1
        self.max_samples_1 = max_samples_1
        self.n_estimators_2 = n_estimators_2
        self.max_samples_2 = max_samples_2
        self.random_state = random_state
        self.contamination = contamination
        self.method = method

    def fit(self, X):
        """Fit the IKGAD model.

        Parameters
        ----------
        X : array-like of shape (n_groups, n_samples, n_features)
            The input data, where n_groups is the number of groups,
            n_samples is the number of instances per group, and
            n_features is the number of features.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Sets the `is_fitted_` attribute to `True`.
        """
        # Validate input data
        X = check_format(X)
        # Fit the model
        self._fit(X)
        self.is_fitted_ = True

        # Set threshold
        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    f"contamination must be in (0, 0.5], got: {self.contamination}"
                )
            # Define threshold based on contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(X), 100.0 * self.contamination
            )
        else:
            # Use default threshold as described in the original paper
            self.offset_ = -0.5

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
        """Internal method to fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_groups, n_samples, n_features)
            Training data
        """
        # Flatten all groups into a single dataset for the first isolation kernel
        X_full = np.vstack(X)

        # First level isolation kernel
        iso_kernel_1 = IsoKernel(
            n_estimators=self.n_estimators_1,
            max_samples=self.max_samples_1,
            random_state=self.random_state,
            method=self.method,
        )
        self.iso_kernel_1_ = iso_kernel_1.fit(X_full)

        return self

    def predict(self, X):
        """Predict if groups are outliers or inliers.

        Parameters
        ----------
        X : array-like of shape (n_groups, n_samples, n_features)
            The input groups to evaluate

        Returns
        -------
        is_inlier : ndarray of shape (n_groups,)
            For each group, returns whether it is an inlier (+1) or
            outlier (-1) according to the fitted model.
        """
        check_is_fitted(self, "is_fitted_")
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """Compute decision scores for groups.

        Parameters
        ----------
        X : array-like of shape (n_groups, n_samples, n_features)
            The input groups to evaluate

        Returns
        -------
        scores : ndarray of shape (n_groups,)
            Decision scores. Negative scores represent outliers,
            positive scores represent inliers.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """Compute anomaly scores for groups.

        Parameters
        ----------
        X : array-like of shape (n_groups, n_samples, n_features)
            The input groups to evaluate

        Returns
        -------
        scores : ndarray of shape (n_groups,)
            Anomaly scores where lower values indicate more anomalous groups.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_format(X)

        X_full = np.vstack(X)
        # Create kernel mean embeddings for each group
        split_idx = np.cumsum([len(x) for x in X])
        X_trans = self.iso_kernel_1_.transform(X_full)
        group_embeddings = np.split(X_trans.toarray(), split_idx[:-1], axis=0)
        X_embeddings = np.asarray(
            [
                self._kernel_mean_embedding(x, self.n_estimators_1)
                for x in group_embeddings
            ]
        )

        # Second level isolation kernel on the embeddings
        iso_kernel_2 = IsoKernel(
            n_estimators=self.n_estimators_2,
            max_samples=self.max_samples_2,
            random_state=self.random_state,
            method=self.method,
        )

        X_trans = iso_kernel_2.fit_transform(X_embeddings)
        kme = self._kernel_mean_embedding(X_trans, self.n_estimators_2)

        # For sparse matrices, .A1 converts to 1D array
        if hasattr(X_trans, "A1"):
            scores = safe_sparse_dot(X_trans, kme.T).A1
        else:
            scores = safe_sparse_dot(X_trans, kme.T)
            if hasattr(scores, "A1"):
                scores = scores.A1
            elif scores.ndim > 1:
                scores = scores.ravel()

        return -scores
