"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import warnings
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from pyikt.group import IKGAD


class IKTOD(OutlierMixin, BaseEstimator):
    """Isolation Kernel-based Time series Subsequnce Anomaly Detection.

    IKTOD implements a distribution-based approach for anomaly time series subsequence detection.
    Unlike traditional time or frequency domain approaches that rely on sliding windows,
    IKTOD treats time series subsequences as distributions in R domain, enabling more
    effective similarity measurements with linear time complexity.

    This approach uses Isolation Distributional Kernel (IDK) to measure similarities
    between subsequences, resulting in better detection accuracy compared to
    sliding-window-based detectors.

    Parameters
    ----------
    n_estimators_1 : int, default=100
        Number of base estimators in the first-level ensemble.

    max_samples_1 : int, float, or "auto", default="auto"
        Number of samples for training each first-level base estimator:
        - int: exactly `max_samples_1` samples
        - float: `max_samples_1 * X.shape[0]` samples
        - "auto": `min(8, n_samples)`

    n_estimators_2 : int, default=100
        Number of base estimators in the second-level ensemble.

    max_samples_2 : int, float, or "auto", default="auto"
        Number of samples for training each second-level base estimator:
        - int: exactly `max_samples_2` samples
        - float: `max_samples_2 * X.shape[0]` samples
        - "auto": `min(8, n_samples)`

    method : {"inne", "anne"}, default="inne"
        Isolation method to use:
        - "inne": original Isolation Forest approach
        - "anne": approximate nearest neighbor ensemble

    period_length : int, default=10
        Length of subsequences to split the time series.

    contamination : "auto" or float, default="auto"
        Proportion of outliers in the dataset:
        - "auto": threshold determined as in the original paper
        - float: must be in range (0, 0.5]

    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.

    Attributes
    ----------
    ikgad_ : IKGAD
        Trained Isolation Kernel Group Anomaly Detector.

    offset_ : float
        Decision threshold for outlier detection.

    is_fitted_ : bool
        Indicates if the model has been fitted.

    References
    ----------
    .. [1] Ting, K.M., Liu, Z., Zhang, H., Zhu, Y. (2022). A New Distributional
           Treatment for Time Series and An Anomaly Detection Investigation.
           Proceedings of The Very Large Data Bases (VLDB) Conference.

    Examples
    --------
    >>> from pyikt.timeseries import IKTOD
    >>> import numpy as np
    >>> # Time series with length 40 (4 periods of length 10)
    >>> X = np.sin(np.linspace(0, 8*np.pi, 40)).reshape(-1, 1)
    >>> # Add anomaly
    >>> X[25:30] = X[25:30] + 5.0
    >>> detector = IKTOD(max_samples_1=2, max_samples_2=2, contamination=0.1, random_state=42)
    >>> detector = detector.fit(X)
    >>> detector.predict(X)
    array([ 1,  1, -1,  1])
    """

    def __init__(
        self,
        n_estimators_1: int = 100,
        max_samples_1: Union[int, float, str] = "auto",
        n_estimators_2: int = 100,
        max_samples_2: Union[int, float, str] = "auto",
        method: str = "inne",
        period_length: int = 10,
        contamination: Union[str, float] = "auto",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_estimators_1 = n_estimators_1
        self.max_samples_1 = max_samples_1
        self.n_estimators_2 = n_estimators_2
        self.max_samples_2 = max_samples_2
        self.period_length = period_length
        self.random_state = random_state
        self.contamination = contamination
        self.method = method

    def fit(self, X) -> "IKTOD":
        """Fit the IKTOD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input time series data where:
            - n_samples: length of the time series
            - n_features: number of variables (default 1 for univariate)

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If time series length is less than or equal to period_length.
        """
        # Validate input data
        X = check_array(X)

        if len(X) <= self.period_length:
            raise ValueError(
                f"Time series length ({X.shape[0]}) must be greater than "
                f"period_length ({self.period_length})."
            )

        # Check if time series length is compatible with period_length
        rest_samples = X.shape[0] % self.period_length
        if rest_samples != 0:
            warnings.warn(
                f"The last sequence of series has {rest_samples} samples, "
                f"which are less than other sequence."
            )

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
            scores = self.score_samples(X)
            self.offset_ = np.percentile(scores, 100.0 * self.contamination)
        else:
            # Use default threshold as described in the original paper
            self.offset_ = -0.5

        return self

    def _fit(self, X) -> "IKTOD":
        """Internal method to fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Time series data

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Split time series into subsequences
        X_sep = self._split_to_subsequences(X)

        # Initialize and fit IKGAD
        self.ikgad_ = IKGAD(
            n_estimators_1=self.n_estimators_1,
            max_samples_1=self.max_samples_1,
            n_estimators_2=self.n_estimators_2,
            max_samples_2=self.max_samples_2,
            method=self.method,
            contamination=self.contamination,
            random_state=self.random_state,
        )

        self.ikgad_.fit(X_sep)
        return self

    def _split_to_subsequences(self, X) -> np.ndarray:
        """Split time series into subsequences based on period_length.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Time series data

        Returns
        -------
        X_sep : ndarray
            Subsequences of the original time series
        """

        indices = np.arange(self.period_length, X.shape[0], self.period_length)
        return np.array_split(X, indices, 0)

    def predict(self, X) -> np.ndarray:
        """Predict if subsequences contain outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Time series data to evaluate

        Returns
        -------
        labels : ndarray of shape (n_subsequences,)
            Returns +1 for inliers and -1 for outliers for each subsequence.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        X_sep = self._split_to_subsequences(X)
        return self.ikgad_.predict(X_sep)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision scores for subsequences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Time series data to evaluate

        Returns
        -------
        scores : ndarray of shape (n_subsequences,)
            Decision scores. Negative scores represent outliers,
            positive scores represent inliers.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X) -> np.ndarray:
        """Compute anomaly scores for subsequences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Time series data to evaluate

        Returns
        -------
        scores : ndarray of shape (n_subsequences,)
            Anomaly scores where lower values indicate more anomalous subsequences.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        X_sep = self._split_to_subsequences(X)
        return self.ikgad_.score_samples(X_sep)
