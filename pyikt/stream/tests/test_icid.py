# Copyright 2024 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
import pytest
from sklearn.utils import check_random_state

from pyikt.stream import ICID

rng = check_random_state(42)


def test_icid_init():
    """Test ICID initialization with default parameters."""
    icid = ICID()
    assert icid.n_estimators == 200
    assert icid.max_samples_list == [2, 4, 8, 16, 32, 64]
    assert icid.method == "inne"
    assert icid.stability_method == "entropy"
    assert icid.contamination == "auto"
    assert icid.window_size == 10
    assert icid.adjust_rate == 0.1


def test_icid_fit():
    """Test ICID fit method."""
    X = np.random.randn(100, 2)
    icid = ICID(n_estimators=50, max_samples_list=[4, 8], random_state=42)
    icid.fit(X)
    assert hasattr(icid, "is_fitted_")
    assert hasattr(icid, "best_iso_kernel_")
    assert hasattr(icid, "best_stability_score_")
    assert hasattr(icid, "interval_score_")
    assert hasattr(icid, "pre_interval_")


def test_icid_fit_predict_batch():
    """Test ICID fit_predict_batch method."""
    # Create a dataset with distribution change
    np.random.seed(42)
    X_normal1 = np.random.randn(30, 2)
    X_anomaly = np.random.randn(10, 2) * 5 + 10  # Different distribution
    X_normal2 = np.random.randn(30, 2)
    X = np.vstack([X_normal1, X_anomaly, X_normal2])

    icid = ICID(
        n_estimators=50, max_samples_list=[4, 8], window_size=10, random_state=42
    )
    predictions = icid.fit_predict_batch(X)

    # We should get predictions for intervals, not individual points
    assert len(predictions) <= len(X) // 10  # Number of intervals


def test_icid_predict_online():
    """Test ICID predict_online method."""
    # Initial training data
    X_train = np.random.randn(50, 2)

    # New intervals - one normal, one anomalous
    X_normal = np.random.randn(10, 2)
    X_anomaly = np.random.randn(10, 2) * 5 + 10

    icid = ICID(
        n_estimators=50, max_samples_list=[4, 8], window_size=10, random_state=42
    )
    icid.fit(X_train)

    # Predict on new data
    normal_result = icid.predict_online(X_normal)
    anomaly_result = icid.predict_online(X_anomaly)

    assert normal_result in [1, -1]
    assert anomaly_result in [1, -1]


def test_icid_stability_methods():
    """Test ICID with different stability methods."""
    X = np.random.randn(100, 2)

    stability_methods = ["entropy", "variance", "mean"]
    for method in stability_methods:
        icid = ICID(
            stability_method=method,
            n_estimators=50,
            max_samples_list=[4],
            random_state=42,
        )
        icid.fit(X)
        assert isinstance(icid.best_stability_score_, float)


def test_icid_invalid_stability_method():
    """Test ICID with invalid stability method raises ValueError."""
    X = np.random.randn(100, 2)
    icid = ICID(stability_method="invalid_method")

    with pytest.raises(ValueError):
        icid.fit(X)


def test_icid_distribution_changes():
    """Test ICID detects distribution changes."""
    # Create data with clear distribution shift
    np.random.seed(42)
    X1 = np.random.randn(30, 2)
    X2 = np.random.randn(30, 2) * 2 + 5  # Shifted distribution
    X3 = np.random.randn(30, 2)
    X = np.vstack([X1, X2, X3])

    icid = ICID(
        n_estimators=100,
        max_samples_list=[4, 8],
        window_size=10,
        adjust_rate=0.5,
        random_state=42,
    )
    results = icid.fit_predict_batch(X)

    # We expect some intervals to be detected as changes
    assert -1 in results


def test_icid_different_window_sizes():
    """Test ICID with different window sizes."""
    X = np.random.randn(100, 2)

    window_sizes = [5, 10, 20]
    for window_size in window_sizes:
        icid = ICID(
            window_size=window_size,
            n_estimators=50,
            max_samples_list=[4],
            random_state=42,
        )
        icid.fit(X)
        assert hasattr(icid, "interval_score_")


def test_icid_different_isolation_methods():
    """Test ICID with different isolation methods."""
    X = np.random.randn(100, 2)

    methods = ["inne", "anne"]
    for method in methods:
        icid = ICID(
            method=method, n_estimators=50, max_samples_list=[4], random_state=42
        )
        icid.fit(X)
        assert hasattr(icid, "best_iso_kernel_")
