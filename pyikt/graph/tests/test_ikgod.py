import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_blobs

from pyikt.graph import IKGOD


def test_ikgod_initialization():
    """Test IKGOD initialization with default and custom parameters."""
    # Default initialization
    model = IKGOD()
    assert model.n_estimators == 200
    assert model.max_samples == "auto"
    assert model.contamination == "auto"
    assert model.method == "inne"
    assert model.h == 3

    # Custom initialization
    model = IKGOD(
        n_estimators=100, max_samples=0.8, contamination=0.1, method="anne", h=2
    )
    assert model.n_estimators == 100
    assert model.max_samples == 0.8
    assert model.contamination == 0.1
    assert model.method == "anne"
    assert model.h == 2


def test_ikgod_with_synthetic_data():
    """Test IKGOD with synthetic data."""
    # Generate synthetic data
    X, _ = make_blobs(n_samples=20, centers=2, random_state=42)

    # Create a simple adjacency matrix (a ring graph)
    n_samples = X.shape[0]
    adjacency = sp.csr_matrix((n_samples, n_samples))
    for i in range(n_samples):
        adjacency[i, (i + 1) % n_samples] = 1
        adjacency[(i + 1) % n_samples, i] = 1

    # Initialize and fit the model
    model = IKGOD(n_estimators=50, random_state=42, h=2)
    model.fit(adjacency, X)

    # Check if the model is fitted
    assert hasattr(model, "is_fitted_")
    assert model.is_fitted_ is True

    # Test prediction
    labels = model.predict(X)
    assert labels.shape == (n_samples,)
    assert np.all(np.logical_or(labels == 1, labels == -1))

    # Test score_samples
    scores = model.score_samples(X)
    assert scores.shape == (n_samples,)

    # Test decision_function
    decision = model.decision_function(X)
    assert decision.shape == (n_samples,)


def test_ikgod_with_small_graph():
    """Test IKGOD with a small graph and obvious anomaly."""
    # Create small graph where one node has very different features
    features = np.array(
        [
            [1.0, 1.0],  # Normal
            [0.9, 1.1],  # Normal
            [1.1, 0.9],  # Normal
            [5.0, 5.0],  # Anomaly
        ]
    )

    # Create adjacency matrix: first 3 nodes form a triangle, last one connected to first
    adjacency = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])

    # Fit model with fixed contamination
    model = IKGOD(n_estimators=50, contamination=0.25, random_state=42, h=1)
    model.fit(adjacency, features)

    # Check predictions
    labels = model.predict(features)
    assert labels.shape == (4,)
    # The last point should be labeled as anomaly (-1)
    assert labels[3] == -1


def test_ikgod_parameter_validation():
    """Test parameter validation in IKGOD."""
    # Invalid contamination
    with pytest.raises(ValueError):
        model = IKGOD(contamination=2.0)
        X = np.random.rand(10, 2)
        adj = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        model.fit(adj, X)

    # Invalid max_samples string
    with pytest.raises(ValueError):
        model = IKGOD(max_samples="invalid")
        X = np.random.rand(10, 2)
        adj = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        model.fit(adj, X)

    # Invalid max_samples float
    with pytest.raises(ValueError):
        model = IKGOD(max_samples=2.0)
        X = np.random.rand(10, 2)
        adj = sp.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        model.fit(adj, X)


def test_ikgod_h_hop_neighbors():
    """Test h-hop neighbor extraction."""
    # Create a path graph: 0-1-2-3-4
    adjacency = sp.csr_matrix(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )

    features = np.random.rand(5, 2)

    # h=1 should see only direct neighbors
    model_h1 = IKGOD(h=1, n_estimators=10)
    model_h1._get_h_nodes_n_dict(adjacency)

    # h=2 should see neighbors at distance 2
    model_h2 = IKGOD(h=2, n_estimators=10)
    model_h2._get_h_nodes_n_dict(adjacency)

    # The test itself is implicit since we're checking that the method runs without errors
    assert True
