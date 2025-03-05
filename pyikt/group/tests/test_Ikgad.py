"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
import pytest

from pyikt.group import IKGAD


def test_IKGAD_initialization():
    clf = IKGAD(
        n_estimators_1=100,
        max_samples_1=0.5,
        n_estimators_2=150,
        max_samples_2=0.8,
        method="inne",
        contamination=0.1,
        random_state=42,
    )

    assert clf.n_estimators_1 == 100
    assert clf.max_samples_1 == 0.5
    assert clf.n_estimators_2 == 150
    assert clf.max_samples_2 == 0.8
    assert clf.method == "inne"
    assert clf.contamination == 0.1
    assert clf.random_state == 42


def test_IKGAD_fit():
    # Create a sample dataset with 2D features in each group
    X = np.array(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]]
    )
    clf = IKGAD()
    clf.fit(X)
    assert clf.is_fitted_
    assert hasattr(clf, "iso_kernel_1_")
    assert hasattr(clf, "offset_")


def test_IKGAD_predict():
    X = [
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
        [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]],
    ]
    clf = IKGAD(random_state=42)
    clf.fit(X)

    predictions = clf.predict(X)

    assert predictions.shape == (3,)
    assert np.all(np.isin(predictions, [-1, 1]))


def test_IKGAD_decision_function():
    X = np.array(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]]
    )
    clf = IKGAD(random_state=42)
    clf.fit(X)

    decision_scores = clf.decision_function(X)

    assert decision_scores.shape == (2,)


def test_IKGAD_score_samples():
    X = np.array(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]]
    )
    clf = IKGAD(random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)

    assert scores.shape == (2,)


def test_IKGAD_with_contamination_parameter():
    X = [
        [[1.0, 1.1], [1.2, 1.3]],
        [[1.3, 1.2], [1.1, 1.0]],
        [[1.0, 1.2], [1.4, 1.3]],
        [[5.0, 5.1], [5.2, 5.3]],
    ]

    # Test with explicit contamination parameter
    clf = IKGAD(max_samples_1=2, max_samples_2=2, contamination=0.25, random_state=42)
    clf.fit(X)

    # Should mark one group as anomaly (25% contamination)
    predictions = clf.predict(X)
    assert sum(predictions == -1) == 1


def test_IKGAD_input_validation():
    # Test with invalid dimensions
    X_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    clf = IKGAD()

    with pytest.raises(ValueError):
        clf.fit(X_2d)

    # Test with empty groups
    X_empty_group = np.array([[[]]])
    with pytest.raises(ValueError):
        clf.fit(X_empty_group)

    # Test with invalid contamination
    with pytest.raises(ValueError):
        IKGAD(contamination=0.6).fit(np.array([[[1.0, 2.0]], [[2.0, 3.0]]]))


def test_IKGAD_list_input():
    # Test with list input instead of numpy array
    X_list = [[[1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [4.0, 5.0]]]
    clf = IKGAD(random_state=42)
    clf.fit(X_list)

    scores = clf.score_samples(X_list)
    assert scores.shape == (2,)


def test_IKGAD_predict_new_data():
    # Train on some data
    X_train = np.array(
        [
            [[1.0, 1.1], [1.2, 1.3], [0.8, 0.9]],
            [[0.9, 0.8], [1.1, 1.0], [1.0, 1.2]],
            [[5.0, 5.1], [5.2, 5.3], [4.9, 4.8]],
        ]
    )
    clf = IKGAD(random_state=42)
    clf.fit(X_train)

    # Predict on new data
    X_test = np.array([[[1.1, 1.0], [0.9, 1.1]], [[5.1, 5.0], [4.8, 4.9]]])
    predictions = clf.predict(X_test)

    assert predictions.shape == (2,)
