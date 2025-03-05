"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from pyikt.anomaly import IDKD

method = ["inne", "anne"]


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("method", method)
def test_IDKD_fit(data, method):
    X = data[0]
    idkd = IDKD(method=method, n_estimators=200)
    idkd.fit(X)
    assert idkd.is_fitted_


@pytest.mark.parametrize("method", method)
def test_IDKD_predict(data, method):
    X = data[0]
    idkd = IDKD(method=method, n_estimators=200)
    idkd.fit(X)
    # Test prediction on training data
    predictions = idkd.predict(X[:10])
    assert predictions.shape == (10,)
    assert np.all(np.isin(predictions, [-1, 1]))


@pytest.mark.parametrize("method", method)
def test_IDKD_decision_function(data, method):
    X = data[0]
    idkd = IDKD(method=method, n_estimators=200)
    idkd.fit(X)
    # Test decision function
    decision_scores = idkd.decision_function(X[:10])
    assert decision_scores.shape == (10,)


@pytest.mark.parametrize("method", method)
def test_IDKD_score_samples(data, method):
    X = data[0]
    idkd = IDKD(method=method, n_estimators=200)
    idkd.fit(X)
    # Test score samples
    scores = idkd.score_samples(X[:10])
    assert scores.shape == (10,)


@pytest.mark.parametrize("contamination", [0.1, 0.2, 0.3, "auto"])
def test_IDKD_contamination(data, contamination):
    X = data[0]
    idkd = IDKD(contamination=contamination, n_estimators=100)
    idkd.fit(X)
    assert idkd.is_fitted_


def test_IDKD_invalid_contamination(data):
    X = data[0]
    idkd = IDKD(contamination=0.6)  # Invalid: > 0.5
    with pytest.raises(ValueError):
        idkd.fit(X)


@pytest.mark.parametrize("max_samples", [10, 20, "auto"])
def test_IDKD_max_samples(data, max_samples):
    X = data[0]
    idkd = IDKD(max_samples=max_samples, n_estimators=100)
    idkd.fit(X)
    assert idkd.is_fitted_


def test_IDKD_invalid_max_samples():
    with pytest.raises(ValueError):
        idkd = IDKD(max_samples="invalid")
        idkd.fit(np.random.random((10, 2)))


def test_IDKD_simple_example():
    X = np.array([[-1.1, 0.2], [0.3, 0.5], [0.5, 1.1], [100, 90]])
    clf = IDKD(max_samples=2, contamination=0.25).fit(X)
    predictions = clf.predict([[0.1, 0.3], [0, 0.7], [90, 85]])
    # The sample close to 100 should be classified as an outlier (-1)
    assert predictions[-1] == -1
    # The first two samples should be inliers (1)
    assert predictions[0] == 1
    assert predictions[1] == 1
