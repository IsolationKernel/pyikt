"""pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
import pytest

from pyikt.cluster import IKAHC


def test_ikahc_initialization():
    """Test IKAHC initialization with various parameters."""
    # Default initialization
    ikahc = IKAHC()
    assert ikahc.n_estimators == 200
    assert ikahc.max_samples == "auto"
    assert ikahc.ik_method == "anne"
    assert ikahc.lk_method == "single"
    assert ikahc.return_flat is False
    assert ikahc.t is None
    assert ikahc.n_clusters is None
    assert ikahc.criterion == "distance"
    assert ikahc.random_state is None

    # Custom initialization
    ikahc = IKAHC(
        n_estimators=100,
        max_samples=5,
        lk_method="complete",
        ik_method="inne",
        return_flat=True,
        t=0.5,
        criterion="distance",
        random_state=42,
    )
    assert ikahc.n_estimators == 100
    assert ikahc.max_samples == 5
    assert ikahc.ik_method == "inne"
    assert ikahc.lk_method == "complete"
    assert ikahc.return_flat is True
    assert ikahc.t == 0.5
    assert ikahc.n_clusters is None
    assert ikahc.criterion == "distance"
    assert ikahc.random_state == 42


def test_ikahc_fit():
    """Test IKAHC fit method."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
    ikahc = IKAHC(n_estimators=100, max_samples=2, random_state=42)

    result = ikahc.fit(X)

    # Check if fit returns self
    assert result is ikahc

    # Check if attributes are set
    assert hasattr(ikahc, "isokernel_")
    assert hasattr(ikahc, "dendrogram_")

    # Check dendrogram shape
    assert ikahc.dendrogram_.shape == (3, 4)  # (n_samples-1) x 4


def test_ikahc_fit_with_return_flat():
    """Test IKAHC fit method with return_flat=True."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])

    # Test with t parameter
    ikahc = IKAHC(
        n_estimators=100, max_samples=2, return_flat=True, t=0.5, random_state=42
    )
    ikahc.fit(X)
    assert hasattr(ikahc, "labels_")
    assert ikahc.labels_ is not None

    # Test with n_clusters parameter
    ikahc = IKAHC(
        n_estimators=100, max_samples=2, return_flat=True, n_clusters=2, random_state=42
    )
    ikahc.fit(X)
    assert hasattr(ikahc, "labels_")
    assert ikahc.labels_ is not None


def test_ikahc_extract_flat_cluster():
    """Test IKAHC _extract_flat_cluster method."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
    ikahc = IKAHC(n_estimators=100, max_samples=2, random_state=42)
    ikahc.fit(X)

    # Test with distance threshold
    labels_t = ikahc._extract_flat_cluster(t=0.5)
    assert labels_t.shape == (4,)

    # Test with n_clusters
    labels_n = ikahc._extract_flat_cluster(n_clusters=2)
    assert labels_n.shape == (4,)

    # Test error cases
    with pytest.raises(ValueError):
        ikahc._extract_flat_cluster(t=0.5, n_clusters=2)

    with pytest.raises(ValueError):
        ikahc._extract_flat_cluster()


def test_ikahc_properties():
    """Test IKAHC properties."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
    ikahc = IKAHC(n_estimators=100, max_samples=2, random_state=42)
    ikahc.fit(X)

    # Test dendrogram property
    dendrogram = ikahc.dendrogram
    assert np.array_equal(dendrogram, ikahc.dendrogram_)

    # Test isokernel property
    isokernel = ikahc.isokernel
    assert isokernel is ikahc.isokernel_


def test_ikahc_fit_transform():
    """Test IKAHC fit_transform method."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
    ikahc = IKAHC(n_estimators=100, max_samples=2, random_state=42)

    dendrogram = ikahc.fit_transform(X)

    # Check if dendrogram is returned
    assert dendrogram.shape == (3, 4)

    # Check if it's the same as dendrogram_
    assert np.array_equal(dendrogram, ikahc.dendrogram_)


def test_ikahc_fit_predict():
    """Test IKAHC fit_predict method."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
    ikahc = IKAHC(
        n_estimators=100, max_samples=2, random_state=42, return_flat=True, n_clusters=2
    )

    labels = ikahc.fit_predict(X)
    assert labels.shape == (4,)
    assert isinstance(labels, np.ndarray)


def test_ikahc_invalid_parameters():
    """Test IKAHC with invalid parameters."""
    X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])

    # Invalid ik_method
    ikahc = IKAHC(ik_method="invalid")
    with pytest.raises(ValueError):
        ikahc.fit(X)

    # Invalid lk_method
    ikahc = IKAHC(lk_method="invalid")
    with pytest.raises(ValueError):
        ikahc.fit(X)

    # Invalid n_estimators
    ikahc = IKAHC(n_estimators=0)
    with pytest.raises(ValueError):
        ikahc.fit(X)

    # Both t and n_clusters provided
    ikahc = IKAHC(return_flat=True, t=0.5, n_clusters=2)
    with pytest.raises(ValueError):
        ikahc.fit(X)
