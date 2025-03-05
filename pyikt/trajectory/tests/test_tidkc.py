import numpy as np
import pytest

from pyikt.trajectory import TIDKC


def test_tidkc_initialization():
    """Test TIDKC initialization with default parameters."""
    tidkc = TIDKC(k=2, kn=5, v=0.5, n_init_samples=10)
    assert tidkc.k == 2
    assert tidkc.kn == 5
    assert tidkc.v == 0.5
    assert tidkc.n_init_samples == 10
    assert tidkc.method == "anne"
    assert tidkc.is_post_process is True


def test_tidkc_invalid_method():
    """Test TIDKC with invalid method parameter."""
    tidkc = TIDKC(k=2, kn=5, v=0.5, n_init_samples=10, method="invalid_method")

    # Create some simple trajectory data
    X = [np.random.rand(5, 2) for _ in range(10)]

    with pytest.raises(ValueError, match="method must be one of 'inne', 'anne'"):
        tidkc.fit(X)


def test_tidkc_fit_predict():
    """Test TIDKC fit and predict functionality."""
    # Create sample trajectory data
    n_trajectories = 20
    trajectory_length = 10
    n_features = 2

    # Create two clusters of trajectories
    X = []
    for i in range(n_trajectories):
        if i < n_trajectories // 2:
            # Cluster 1: trajectories in the top half of the space
            trajectory = np.random.rand(trajectory_length, n_features)
            trajectory[:, 1] += 1  # Move up in y-dimension
        else:
            # Cluster 2: trajectories in the bottom half of the space
            trajectory = np.random.rand(trajectory_length, n_features)
        X.append(trajectory)

    # Fit the model with a small number of estimators for quick testing
    tidkc = TIDKC(
        k=2,
        kn=3,
        v=0.5,
        n_init_samples=5,
        n_estimators_1=100,
        max_samples_1=2,
        n_estimators_2=100,
        max_samples_2=2,
        random_state=42,
    )

    # Test fit
    tidkc.fit(X)
    assert hasattr(tidkc, "labels_")
    assert hasattr(tidkc, "idkc_")

    # Test predict
    labels = tidkc.fit_predict(X)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (n_trajectories,)
    assert len(np.unique(labels)) <= 2  # Should have at most 2 clusters


def test_tidkc_with_different_parameters():
    """Test TIDKC with different parameters."""
    # Create simple trajectory data
    np.random.seed(42)
    X = [np.random.rand(5, 2) for _ in range(15)]

    # Test with different method
    tidkc1 = TIDKC(
        k=3,
        kn=4,
        v=0.4,
        n_init_samples=5,
        method="anne",
        n_estimators_1=100,
        max_samples_1=2,
        n_estimators_2=100,
        max_samples_2=2,
        random_state=42,
    )
    labels1 = tidkc1.fit_predict(X)

    # Test with no post-processing
    tidkc2 = TIDKC(
        k=3,
        kn=4,
        v=0.4,
        n_init_samples=5,
        is_post_process=False,
        n_estimators_1=100,
        max_samples_1=2,
        n_estimators_2=100,
        max_samples_2=2,
        random_state=42,
    )
    labels2 = tidkc2.fit_predict(X)

    assert labels1.shape == (len(X),)
    assert labels2.shape == (len(X),)
