import numpy as np
import scipy.sparse as sp

from pyikt.graph import IsoGraphKernel


def test_isographkernel_initialization():
    """Test IsoGraphKernel initialization with default and custom parameters."""
    # Default initialization
    igk = IsoGraphKernel()
    assert igk.n_estimators == 200
    assert igk.max_samples == "auto"
    assert igk.method == "anne"
    assert igk.random_state is None

    # Custom initialization
    igk = IsoGraphKernel(
        method="inne", n_estimators=100, max_samples=0.8, random_state=42
    )
    assert igk.n_estimators == 100
    assert igk.max_samples == 0.8
    assert igk.method == "inne"
    assert igk.random_state == 42


def test_isographkernel_fit():
    """Test IsoGraphKernel fit method."""
    # Create sample features
    features = np.random.rand(10, 5)

    # Initialize and fit the model
    igk = IsoGraphKernel(n_estimators=50, random_state=42)
    fitted_model = igk.fit(features)

    # Check if the model is fitted
    assert hasattr(fitted_model, "is_fitted_")
    assert fitted_model.is_fitted_ is True
    assert hasattr(fitted_model, "iso_kernel_")


def test_isographkernel_transform():
    """Test IsoGraphKernel transform method."""
    # Create sample features and adjacency matrix
    features = np.random.rand(10, 5)
    adjacency = np.zeros((10, 10))
    for i in range(10):
        adjacency[i, (i + 1) % 10] = 1
        adjacency[(i + 1) % 10, i] = 1
    adjacency = sp.csr_matrix(adjacency)

    # Initialize, fit and transform
    igk = IsoGraphKernel(n_estimators=50, random_state=42)
    igk.fit(features)

    # Test transform with sparse output
    embedding_sparse = igk.transform(adjacency, features, h=1, dense_output=False)
    assert sp.issparse(embedding_sparse)

    # Test transform with dense output
    embedding_dense = igk.transform(adjacency, features, h=1, dense_output=True)
    assert isinstance(embedding_dense, np.ndarray)


def test_isographkernel_fit_transform():
    """Test IsoGraphKernel fit_transform method."""
    # Create sample features and adjacency matrix
    features = np.random.rand(10, 5)
    adjacency = np.zeros((10, 10))
    for i in range(10):
        adjacency[i, (i + 1) % 10] = 1
        adjacency[(i + 1) % 10, i] = 1
    adjacency = sp.csr_matrix(adjacency)

    # Test fit_transform with sparse output
    igk = IsoGraphKernel(n_estimators=50, random_state=42)
    embedding_sparse = igk.fit_transform(adjacency, features, h=1, dense_output=False)
    assert sp.issparse(embedding_sparse)
    assert igk.is_fitted_ is True

    # Test fit_transform with dense output
    igk = IsoGraphKernel(n_estimators=50, random_state=42)
    embedding_dense = igk.fit_transform(adjacency, features, h=1, dense_output=True)
    assert isinstance(embedding_dense, np.ndarray)
    assert igk.is_fitted_ is True


def test_isographkernel_with_different_h_values():
    """Test IsoGraphKernel with different h values for WL embedding."""
    # Create sample features and adjacency matrix
    features = np.random.rand(10, 5)
    adjacency = np.zeros((10, 10))
    for i in range(10):
        adjacency[i, (i + 1) % 10] = 1
        adjacency[(i + 1) % 10, i] = 1
    adjacency = sp.csr_matrix(adjacency)

    igk = IsoGraphKernel(n_estimators=50, random_state=42)
    igk.fit(features)

    # Test with h=0
    embedding_h0 = igk.transform(adjacency, features, h=0, dense_output=True)

    # Test with h=1
    embedding_h1 = igk.transform(adjacency, features, h=1, dense_output=True)

    # Test with h=2
    embedding_h2 = igk.transform(adjacency, features, h=2, dense_output=True)

    # Each h value should produce a different embedding size
    # For each h, we add another embedding layer
    assert embedding_h1.shape[1] > embedding_h0.shape[1]
    assert embedding_h2.shape[1] > embedding_h1.shape[1]
