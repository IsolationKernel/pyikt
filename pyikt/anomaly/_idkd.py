"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numbers
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from pyikt.kernel import IsoKernel


class IDKD(OutlierMixin, BaseEstimator):
    """Isolation Distributional Kernel for anomaly detection.

    IDKD measures the similarity between distributions to identify anomalies.
    An observation is considered anomalous when its Dirac measure has a low
    similarity with respect to the reference distribution from which the dataset
    was generated.

    This implementation follows the algorithm described in [1]_.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of base estimators in the ensemble.

    max_samples : {"auto", int, float}, default="auto"
        Number of samples to draw from X to train each base estimator.

        - If "auto", then `max_samples=min(8, n_samples)`.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    method : {"inne", "anne", "auto"}, default="inne"
        Isolation method to use. The original algorithm described in [1]_ uses "inne".

    contamination : {"auto", float}, default="auto"
        The proportion of outliers in the data set.

        - If "auto", the threshold is determined as in [1]_.
        - If float, the contamination should be in the range (0, 0.5].

        Used to define the threshold on the decision function.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    offset_ : float
        Offset used to define the decision function from the raw scores.

    max_samples_ : int
        Actual number of samples used.

    iso_kernel_ : IsoKernel
        The fitted isolation kernel.

    References
    ----------
    .. [1] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2022).
       "Isolation Distributional Kernel: A new tool for kernel based point and group anomaly detections."
       IEEE Transactions on Knowledge and Data Engineering.

    Examples
    --------
    >>> from pyikt.anomaly import IDKD
    >>> import numpy as np
    >>> X = np.array([[-1.1, 0.2], [0.3, 0.5], [0.5, 1.1], [100, 90]])
    >>> clf = IDKD(max_samples=2, contamination=0.25).fit(X)
    >>> clf.predict([[0.1, 0.3], [0, 0.7], [90, 85]])
    array([ 1,  1, -1])
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        method="inne",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.method = method

    def fit(self, X, y=None):
        """Fit the IDKD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Use ``dtype=np.float32`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Check data
        X = check_array(X, accept_sparse=False)

        n_samples = X.shape[0]
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(16, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported."
                    'Valid choices are: "auto", int or'
                    "float" % self.max_samples
                )

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        self._fit(X)
        self.is_fitted_ = True

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
                )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(X), 100.0 * self.contamination
            )

        return self

    def _kernel_mean_embedding(self, X):
        return np.mean(X, axis=0) / self.max_samples_

    def _fit(self, X):
        """Build the isolation kernel from the training data."""
        iso_kernel = IsoKernel(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples_,
            random_state=self.random_state,
            method=self.method,
        )
        self.iso_kernel_ = iso_kernel.fit(X)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Predict if samples are outliers or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns +1 for inliers and -1 for outliers.
        """
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """Compute the decision function for each sample.

        The decision function is defined as score_samples(X) - offset_.
        Negative values are considered outliers and positive values are considered inliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision function values for each sample.
            Negative values represent outliers, positive values represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier.
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """Compute the anomaly scores for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of each input sample.
            The lower the score, the more anomalous the sample.
        """
        check_is_fitted(self, "is_fitted_")
        # Check data
        X = check_array(X, accept_sparse=False)

        X_trans = self.iso_kernel_.transform(X)
        kme = np.average(X_trans.toarray(), axis=0) / self.max_samples_
        scores = safe_sparse_dot(X_trans, kme.T).flatten()

        return scores
