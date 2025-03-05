"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from pyikt.cluster import PSKC


@pytest.fixture
def simple_data():
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    return X


@pytest.fixture
def blob_data():
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    return X, y


def test_pskc_init():
    """Test PSKC initialization with default parameters."""
    pskc = PSKC()
    assert pskc.n_estimators == 200
    assert pskc.max_samples == "auto"
    assert pskc.method == "inne"
    assert pskc.tau == 0.1
    assert pskc.v == 0.1
    assert pskc.random_state is None


def test_pskc_init_custom():
    """Test PSKC initialization with custom parameters."""
    pskc = PSKC(
        n_estimators=100, max_samples=10, method="anne", tau=0.2, v=0.2, random_state=42
    )
    assert pskc.n_estimators == 100
    assert pskc.max_samples == 10
    assert pskc.method == "anne"
    assert pskc.tau == 0.2
    assert pskc.v == 0.2
    assert pskc.random_state == 42


def test_pskc_fit(simple_data):
    """Test PSKC fit method on simple data."""
    X = simple_data
    pskc = PSKC(random_state=42)
    pskc.fit(X)
    assert hasattr(pskc, "is_fitted_")
    assert pskc.is_fitted_
    assert hasattr(pskc, "labels_")
    assert pskc.labels_.shape == (X.shape[0],)
    assert hasattr(pskc, "clusters_")
    assert len(pskc.clusters_) > 0


@pytest.mark.parametrize("method", ["inne", "anne"])
def test_pskc_different_methods(simple_data, method):
    """Test PSKC with different isolation kernel methods."""
    X = simple_data
    pskc = PSKC(method=method, random_state=42)
    pskc.fit(X)
    assert pskc.is_fitted_


def test_pskc_properties(simple_data):
    """Test PSKC properties."""
    X = simple_data
    pskc = PSKC(random_state=42)
    pskc.fit(X)

    # Test clusters property
    assert len(pskc.clusters) > 0

    # Test centers property
    centers = pskc.centers
    assert len(centers) > 0

    # Test n_classes property
    assert pskc.n_classes > 0
    assert pskc.n_classes == len(pskc.clusters)


def test_pskc_blob_data(blob_data):
    """Test PSKC on blob data."""
    X, y = blob_data
    pskc = PSKC(random_state=42)
    pskc.fit(X)
    assert pskc.is_fitted_
    assert pskc.labels_.shape == (X.shape[0],)

    # Typically should find approximately 3 clusters for blobs data
    assert 1 <= pskc.n_classes <= 5


@pytest.mark.parametrize("tau", [0.05, 0.1, 0.2])
def test_pskc_tau_parameter(simple_data, tau):
    """Test PSKC with different tau values."""
    X = simple_data
    pskc = PSKC(tau=tau, random_state=42)
    pskc.fit(X)
    assert pskc.is_fitted_


@pytest.mark.parametrize("v", [0.05, 0.1, 0.2])
def test_pskc_v_parameter(simple_data, v):
    """Test PSKC with different v values."""
    X = simple_data
    pskc = PSKC(v=v, random_state=42)
    pskc.fit(X)
    assert pskc.is_fitted_


@pytest.mark.parametrize("max_samples", [10, 20, "auto"])
def test_pskc_max_samples(simple_data, max_samples):
    """Test PSKC with different max_samples values."""
    X = simple_data
    pskc = PSKC(max_samples=max_samples, random_state=42)
    pskc.fit(X)
    assert pskc.is_fitted_


def test_pskc_reproducibility():
    """Test PSKC reproducibility with fixed random state."""
    X = np.random.rand(20, 2)

    pskc1 = PSKC(random_state=42)
    pskc1.fit(X)
    labels1 = pskc1.labels_

    pskc2 = PSKC(random_state=42)
    pskc2.fit(X)
    labels2 = pskc2.labels_

    np.testing.assert_array_equal(labels1, labels2)
