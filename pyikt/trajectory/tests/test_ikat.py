"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest

from pyikt.trajectory import IKAT


@pytest.fixture
def trajectory_data():
    # Create simple synthetic trajectory data
    # 10 trajectories, each with 5 points in 2D space
    n_trajectories = 10
    n_points = 5
    n_features = 2

    # Generate random trajectories
    np.random.seed(42)
    trajectories = [np.random.rand(n_points, n_features) for _ in range(n_trajectories)]

    # Make one trajectory anomalous by placing it far away
    trajectories[0] = np.random.rand(n_points, n_features) + 10

    return trajectories


@pytest.mark.parametrize("method", ["inne", "anne"])
def test_IKAT_init(method):
    ikat = IKAT(
        n_estimators_1=100,
        max_samples_1=8,
        n_estimators_2=100,
        max_samples_2=8,
        contamination=0.1,
        method=method,
    )
    assert ikat.n_estimators_1 == 100
    assert ikat.max_samples_1 == 8
    assert ikat.n_estimators_2 == 100
    assert ikat.max_samples_2 == 8
    assert ikat.contamination == 0.1
    assert ikat.method == method


def test_IKAT_invalid_method():
    with pytest.raises(ValueError):
        ikat = IKAT(method="invalid_method")
        ikat.fit([np.random.rand(5, 2) for _ in range(10)])


def test_IKAT_invalid_contamination():
    with pytest.raises(ValueError):
        ikat = IKAT(contamination=0.6)
        ikat.fit([np.random.rand(5, 2) for _ in range(10)])


@pytest.mark.parametrize("method", ["inne", "anne"])
def test_IKAT_fit(trajectory_data, method):
    ikat = IKAT(n_estimators_1=50, n_estimators_2=50, method=method, random_state=42)
    ikat.fit(trajectory_data)
    assert hasattr(ikat, "is_fitted_")
    assert ikat.is_fitted_
    assert hasattr(ikat, "ikgad_")
    assert hasattr(ikat, "offset_")


@pytest.mark.parametrize("method", ["inne"])
def test_IKAT_predict(trajectory_data, method):
    ikat = IKAT(
        n_estimators_1=100,
        n_estimators_2=100,
        max_samples_1=2,
        max_samples_2=2,
        method=method,
        contamination=0.1,
        random_state=42,
    )
    ikat.fit(trajectory_data)

    # Test prediction
    predictions = ikat.predict(trajectory_data)
    assert len(predictions) == len(trajectory_data)
    assert set(np.unique(predictions)).issubset({-1, 1})

    # Should have at least one anomaly
    assert -1 in predictions


@pytest.mark.parametrize("method", ["inne", "anne"])
def test_IKAT_score_samples(trajectory_data, method):
    ikat = IKAT(
        n_estimators_1=50,
        n_estimators_2=50,
        method=method,
        contamination="auto",
        random_state=42,
    )
    ikat.fit(trajectory_data)

    # Test scoring
    scores = ikat.score_samples(trajectory_data)
    assert len(scores) == len(trajectory_data)

    # First trajectory should have the lowest score (most anomalous)
    assert scores[0] <= np.min(scores[1:])


@pytest.mark.parametrize("method", ["inne", "anne"])
def test_IKAT_decision_function(trajectory_data, method):
    ikat = IKAT(n_estimators_1=50, n_estimators_2=50, method=method, random_state=42)
    ikat.fit(trajectory_data)

    # Test decision function
    decision_scores = ikat.decision_function(trajectory_data)
    assert len(decision_scores) == len(trajectory_data)

    # Check consistency with predictions
    predictions = ikat.predict(trajectory_data)
    for score, pred in zip(decision_scores, predictions):
        if score < 0:
            assert pred == -1
        else:
            assert pred == 1


def test_IKAT_check_is_fitted():
    ikat = IKAT()
    with pytest.raises(Exception):  # Should raise some kind of not fitted error
        ikat.predict([np.random.rand(5, 2)])
