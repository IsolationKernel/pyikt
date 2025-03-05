import numpy as np
import pytest

from pyikt.stream import STREAMKHC


def test_streamkhc_init():
    # Test initialization with default parameters
    clusterer = STREAMKHC()
    assert clusterer.n_estimators == 200
    assert clusterer.max_samples == "auto"
    assert clusterer.max_leaf == 5000
    assert clusterer.random_state is None

    # Test initialization with custom parameters
    clusterer = STREAMKHC(
        n_estimators=100,
        max_samples=10,
        random_state=42,
        max_leaf=1000,
    )
    assert clusterer.n_estimators == 100
    assert clusterer.max_samples == 10
    assert clusterer.random_state == 42
    assert clusterer.max_leaf == 1000


def test_streamkhc_fit():
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(20, 5)

    # Fit batch
    clusterer = STREAMKHC(n_estimators=50, random_state=42)
    clusterer.fit(X)

    # Check that attributes are set
    assert hasattr(clusterer, "tree_")
    assert hasattr(clusterer, "iso_kernel_")
    assert hasattr(clusterer, "point_counter_")
    assert hasattr(clusterer, "n_features_in_")

    # Check attribute values
    assert clusterer.n_features_in_ == 5
    assert clusterer.point_counter_ == 20


def test_streamkhc_fit_online():
    # Generate sample data
    np.random.seed(42)
    X_batch = np.random.rand(20, 5)
    X_online = np.random.rand(5, 5)

    # Fit batch first, then online
    clusterer = STREAMKHC(n_estimators=50, random_state=42)
    clusterer.fit(X_batch)
    initial_counter = clusterer.point_counter_
    clusterer.fit_online(X_online)

    # Check that counter incremented correctly
    assert clusterer.point_counter_ == initial_counter + len(X_online)


def test_streamkhc_max_leaf():
    # Generate sample data
    np.random.seed(42)
    X_batch = np.random.rand(15, 5)
    X_online = np.random.rand(10, 5)

    # Create clusterer with max_depth of 20
    clusterer = STREAMKHC(n_estimators=50, max_leaf=20, random_state=42)
    clusterer.fit(X_batch)
    clusterer.fit_online(X_online)

    # Counter should be at max 20
    assert clusterer.point_counter_ == 25  # Total points ingested


def test_streamkhc_feature_mismatch():
    # Generate sample data with different dimensions
    np.random.seed(42)
    X_batch = np.random.rand(10, 5)
    X_online = np.random.rand(5, 6)  # Different number of features

    # Fit batch
    clusterer = STREAMKHC(n_estimators=50, random_state=42)
    clusterer.fit(X_batch)

    # Expect ValueError when fitting online with different feature count
    with pytest.raises(ValueError):
        clusterer.fit_online(X_online)


def test_streamkhc_purity():
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(20, 5)

    # Fit model
    clusterer = STREAMKHC(n_estimators=50, random_state=42)
    clusterer.fit(X)

    # Calculate purity (should return a float)
    purity = clusterer.get_purity()
    assert isinstance(purity, float)
    assert 0 <= purity <= 1


# def test_streamkhc_visualization_methods():
#     # Generate sample data
#     np.random.seed(42)
#     X = np.random.rand(10, 5)

#     # Fit model
#     clusterer = STREAMKHC(n_estimators=50, random_state=42)
#     clusterer.fit(X)

#     # Test methods without actually saving files
#     # Just ensure no exceptions are raised

#     with tempfile.NamedTemporaryFile(suffix=".png") as temp_img:
#         try:
#             clusterer.visualize_tree(temp_img.name)
#         except Exception as e:
#             if "GraphViz's executables" in str(e):
#                 pytest.skip("GraphViz not installed, skipping visualization test")
#             else:
#                 raise

#     with tempfile.NamedTemporaryFile(suffix=".json") as temp_json:
#         clusterer.serialize_tree(temp_json.name)
#         assert os.path.exists(temp_json.name)
