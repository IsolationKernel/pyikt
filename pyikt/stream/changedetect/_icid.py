"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyikt.kernel import IsoDisKernel


class ICID(BaseEstimator):
    """Isolate Change Interval Detection for monitoring data stream distribution changes.

    ICID (Isolate Change Interval Detection) is designed to detect intervals in a data
    stream where significant distribution changes occur. It leverages isolation-based
    methods to measure similarity between consecutive data windows, identifying points
    where the underlying distribution shifts. The algorithm adaptively selects the best
    sampling parameters for isolation kernels based on stability metrics.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the isolation distribution kernel.

    max_samples_list : list of int, default=[2, 4, 8, 16, 32, 64]
        List of candidate values for max_samples parameter. The algorithm will
        select the value that yields the most stable isolation kernel.

    method : {'inne', 'anne'}, default='inne'
        The isolation method to use for the kernel.

            - 'inne': Isolation-based Nearest Neighbor Ensemble
            - 'anne': Approximate Nearest Neighbor Ensemble

    stability_method : {'entropy', 'variance', 'mean'}, default='entropy'
        Method used to evaluate the stability of interval scores.

            - 'entropy': Use information entropy as stability measure
            - 'variance': Use variance as stability measure
            - 'mean': Use mean value as stability measure

    window_size : int, default=10
        The size of the sliding window for batch detection.

    adjust_rate : float, default=0.1
        Rate to adjust the threshold for anomaly detection based on
        standard deviation of interval scores.

    contamination : 'auto' or float, default='auto'
        The proportion of outliers in the data set. Used when fitting to define
        the threshold on interval scores.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Attributes
    ----------
    best_iso_kernel_ : IsoDisKernel
        The fitted isolation kernel with the best stability score.

    best_stability_score_ : float
        The stability score of the best isolation kernel.

    interval_score_ : array-like of shape (n_intervals,)
        The dissimilarity scores between consecutive intervals.

    best_max_samples_ : int
        The max_samples parameter of the best isolation kernel.

    pre_interval_ : array-like
        The last interval from the training data, used for online prediction.

    References
    ----------
    .. [1] Y. Cao, Y. Zhu, K. M. Ting, F. D. Salim, H. X. Li, L. Yang, G. Li (2024).
           Detecting change intervals with isolation distributional kernel.
           Journal of Artificial Intelligence Research, 79:273–306.

    Examples
    --------
    >>> from pyikt.stream import ICID
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X_normal1 = np.random.randn(50, 2)
    >>> X_anomaly = np.random.randn(10, 2) * 5 + 10  # Different distribution
    >>> X_normal2 = np.random.randn(20, 2)
    >>> X = np.vstack([X_normal1, X_anomaly, X_normal2])
    >>> icid = ICID(n_estimators=50, max_samples_list=[4, 8], window_size=10, random_state=42)
    >>> # Batch predictions
    >>> icid.fit_predict_batch(X)
    array([ 1,  1,  1,  1,  -1,  -1,  1])
    >>> X_anomaly = np.random.randn(10, 2) * 5 + 10
    >>> X_normal = np.random.randn(10, 2)
    >>> # Predict on new data online
    >>> icid.predict_online(X_normal)
    1
    >>> icid.predict_online(X_anomaly)
    -1
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples_list=[2, 4, 8, 16, 32, 64],
        method="inne",
        stability_method="entropy",
        adjust_rate=0.1,
        contamination="auto",
        window_size=10,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples_list = max_samples_list
        self.method = method
        self.stability_method = stability_method
        self.contamination = contamination
        self.window_size = window_size
        self.random_state = random_state
        self.adjust_rate = adjust_rate
        self.best_iso_kernel_ = None
        self.pre_interval_ = None
        self.interval_score_ = None
        self.best_stability_score_ = float("inf")

    def fit(self, X, y=None):
        """Fit the model on data X in batch mode.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """
        X = check_array(X)
        for max_samples in self.max_samples_list:
            isodiskernel = IsoDisKernel(
                n_estimators=self.n_estimators,
                max_samples=max_samples,
                random_state=self.random_state,
                method=self.method,
            )
            isodiskernel.fit(X)
            interval_scores = self._interval_score(X, isodiskernel, self.window_size)
            stability_score = self._stability_score(interval_scores)
            if stability_score < self.best_stability_score_:
                self.best_iso_kernel_ = isodiskernel
                self.best_stability_score_ = stability_score
                self.interval_score_ = interval_scores
        self.is_fitted_ = True
        return self

    def fit_predict_batch(self, X):
        """Fit the model on data X and predict anomalies in batch mode.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        window_size : int, default=10
            The size of the sliding window.

        Returns
        -------
        is_inlier : np.array of shape (n_intervals,)
            Returns 1 for inliers and -1 for outliers.
        """
        self.fit(X)
        is_inlier = np.ones(len(self.interval_score_), dtype=int)
        threshold = self._determine_anomaly_bounds()
        is_inlier[
            self.interval_score_ > threshold
        ] = -1  # Higher scores indicate change
        return is_inlier

    def predict_online(self, X):
        """Predict if the new data represents a change from the previous interval.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The new data interval to evaluate.

        Returns
        -------
        int : 1 for normal (inlier), -1 for change detected (outlier)
        """
        check_is_fitted(self, ["best_iso_kernel_", "pre_interval_", "interval_score_"])
        X = check_array(X)
        anomaly_score = 1.0 - self.best_iso_kernel_.similarity(self.pre_interval_, X)
        self.interval_score_.append(anomaly_score)
        self.pre_interval_ = X

        threshold = self._determine_anomaly_bounds()
        return 1 if anomaly_score <= threshold else -1

    @property
    def best_stability_score(self):
        """Get the best stability score found during fitting."""
        check_is_fitted(self, ["best_stability_score_"])
        return self.best_stability_score_

    @property
    def best_iso_kernel(self):
        """Get the isolation kernel with the best stability."""
        check_is_fitted(self, ["best_iso_kernel_"])
        return self.best_iso_kernel_

    @property
    def best_max_samples(self):
        """Get the max_samples parameter of the best isolation kernel."""
        check_is_fitted(self, ["best_iso_kernel_"])
        return self.best_iso_kernel_.max_samples

    def _stability_score(self, scores):
        """Calculate stability score based on the chosen method.

        Parameters
        ----------
        scores : np.array
            Array of interval scores.

        Returns
        -------
        float : stability score
        """
        if len(scores) <= 1:
            return float("inf")

        if self.stability_method == "entropy":
            # Normalize scores for entropy calculation
            norm_scores = np.array(scores) / np.sum(scores)
            return entropy(norm_scores)
        elif self.stability_method == "variance":
            return np.var(scores)
        elif self.stability_method == "mean":
            return np.mean(scores)
        else:
            raise ValueError(f"Unknown stability method: {self.stability_method}")

    def _interval_score(self, X, isodiskernel, window_size):
        """Calculate dissimilarity scores between consecutive intervals.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        isodiskernel : IsoDisKernel
            The isolation kernel to use.
        window_size : int
            The size of the sliding window.

        Returns
        -------
        interval_scores : list of float
            Dissimilarity scores between consecutive intervals.
        """
        n_samples = X.shape[0]
        batch_X = [
            X[i : i + window_size]
            for i in range(0, n_samples - window_size + 1, window_size)
        ]

        if len(batch_X) <= 1:
            return []

        interval_scores = []
        for i in range(len(batch_X) - 1):
            interval_scores.append(
                1.0 - isodiskernel.similarity(batch_X[i], batch_X[i + 1])
            )
        self.pre_interval_ = batch_X[-1]
        return interval_scores

    def _determine_anomaly_bounds(self):
        """Determine the threshold for anomaly detection.

        Returns
        -------
        float : threshold value
        """
        mean_score = np.mean(self.interval_score_)
        std_score = np.std(self.interval_score_)
        threshold = mean_score + self.adjust_rate * std_score
        return threshold
