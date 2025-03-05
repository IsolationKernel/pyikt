import numpy as np
import pytest

from pyikt.timeseries import IKTOD


def test_iktod_fit_and_predict():
    # Create a simple sinusoidal time series with an anomaly
    length = 40
    period_length = 10
    X = np.sin(np.linspace(0, 8 * np.pi, length)).reshape(-1, 1)

    # Add anomaly
    X[25:30] = X[25:30] + 2.0

    # Initialize and fit the model
    detector = IKTOD(period_length=period_length, contamination=0.1, random_state=42)

    detector.fit(X)

    # Check if the model is fitted
    assert hasattr(detector, "is_fitted_")
    assert detector.is_fitted_ is True

    # Check if ikgad_ attribute exists
    assert hasattr(detector, "ikgad_")

    # Test predictions
    predictions = detector.predict(X)

    # Should have 4 predictions (one for each subsequence)
    assert len(predictions) == length // period_length

    # Ensure predictions are either -1 or 1
    assert np.all(np.isin(predictions, [-1, 1]))

    # Test decision function
    scores = detector.decision_function(X)
    assert len(scores) == length // period_length

    # Test score_samples
    anomaly_scores = detector.score_samples(X)
    assert len(anomaly_scores) == length // period_length


def test_iktod_invalid_input():
    # Test with incompatible time series length
    length = 5  # Less than period_length=10
    period_length = 10
    X = np.sin(np.linspace(0, 8 * np.pi, length)).reshape(-1, 1)

    detector = IKTOD(period_length=period_length)

    with pytest.raises(
        ValueError, match="Time series length.*must be greater than period_length"
    ):
        detector.fit(X)


def test_iktod_warning_on_misaligned_data():
    # Test warning for misaligned data
    length = 43  # Not divisible by period_length=10
    period_length = 10
    X = np.sin(np.linspace(0, 8 * np.pi, length)).reshape(-1, 1)

    detector = IKTOD(period_length=period_length)

    with pytest.warns(UserWarning, match="The last sequence of series has"):
        detector.fit(X)


def test_iktod_invalid_contamination():
    # Test with invalid contamination value
    X = np.sin(np.linspace(0, 8 * np.pi, 40)).reshape(-1, 1)

    # Contamination > 0.5
    detector = IKTOD(period_length=10, contamination=0.6)

    with pytest.raises(ValueError, match="contamination must be in"):
        detector.fit(X)


def test_iktod_different_methods():
    # Test different isolation methods
    X = np.sin(np.linspace(0, 8 * np.pi, 40)).reshape(-1, 1)

    # Test with 'anne' method
    detector_anne = IKTOD(period_length=10, method="anne", random_state=42)
    detector_anne.fit(X)
    predictions_anne = detector_anne.predict(X)

    # Test with 'inne' method
    detector_inne = IKTOD(period_length=10, method="inne", random_state=42)
    detector_inne.fit(X)
    predictions_inne = detector_inne.predict(X)

    # Both should produce valid predictions
    assert len(predictions_anne) == 4
    assert len(predictions_inne) == 4


def test_iktod_auto_contamination():
    # Test with 'auto' contamination
    X = np.sin(np.linspace(0, 8 * np.pi, 40)).reshape(-1, 1)
    X[25:30] = X[25:30] + 2.0  # Add anomaly

    detector = IKTOD(period_length=10, contamination="auto", random_state=42)
    detector.fit(X)

    # Check if offset_ is set to -0.5 as specified in the code
    assert detector.offset_ == -0.5

    # Test predictions
    predictions = detector.predict(X)
    assert len(predictions) == 4
