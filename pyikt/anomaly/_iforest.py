"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IForest(OutlierMixin, BaseEstimator):
    """Wrapper of scikit-learn Isolation Forest for anomaly detection.

    The IsolationForest 'isolates' observations by randomly selecting a
    feature and then randomly selecting a split value between the maximum and
    minimum values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function. Random partitioning produces
    noticeably shorter paths for anomalies. Hence, when a forest of random trees
    collectively produce shorter path lengths for particular samples, they are
    highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators (trees) in the ensemble.

    max_samples : int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

    contamination : float or 'auto', default=0.1
        The proportion of outliers in the data set. Used to define the threshold
        on the scores of the samples.
        - If 'auto', the threshold is determined as in the original paper.
        - If float, the contamination should be in the range (0, 0.5].

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    detector_ : IsolationForest
        The underlying scikit-learn IsolationForest object.

    is_fitted_ : bool
        Indicates whether the estimator has been fitted.

    References
    ----------
    .. [1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December). "Isolation forest."
           In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

    .. [2] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD), 6(1), 1-39.

    Examples
    --------
    >>> from pyikt.anomaly import IForest
    >>> import numpy as np
    >>> X = np.array([[-1.1, 0.2], [0.3, 0.5], [0.5, 1.1], [100, 90]])
    >>> clf = IForest(contamination=0.25).fit(X)
    >>> clf.predict([[0.1, 0.3], [0, 0.7], [90, 85]])
    array([ 1,  1, -1])
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination=0.1,
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the isolation forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check data
        X = check_array(X, accept_sparse=False)

        self.detector_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.detector_.fit(X=X, y=None, sample_weight=None)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            The predicted labels. +1 for inliers, -1 for outliers.
        """
        check_is_fitted(self, "is_fitted_")
        return self.detector_.predict(X)

    def decision_function(self, X):
        """
        Compute the anomaly score for each sample.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        check_is_fitted(self, "is_fitted_")
        return self.detector_.decision_function(X)

    def score_samples(self, X):
        """
        Return the raw anomaly score of samples.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The raw anomaly score of the input samples.
            The lower, the more abnormal.
        """
        check_is_fitted(self, "is_fitted_")
        # Check data
        X = check_array(X, accept_sparse=False)
        return self.detector_.score_samples(X)
