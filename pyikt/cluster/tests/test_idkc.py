"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs

from pyikt.cluster import IDKC


def test_IDKC():
    # Generate sample data
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    X, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )

    # Initialize IKDC
    n_estimators = 200
    max_samples = 10
    method = "inne"
    k = 3
    kn = 5
    v = 0.1
    n_init_samples = 30
    init_center = None
    is_post_process = True
    random_state = 42
    ikdc = IDKC(
        n_estimators=n_estimators,
        max_samples=max_samples,
        method=method,
        k=k,
        kn=kn,
        v=v,
        n_init_samples=n_init_samples,
        init_center=init_center,
        is_post_process=is_post_process,
        random_state=random_state,
    )

    # Fit IKDC
    labels_pred = ikdc.fit_predict(X)

    # Check if labels are assigned correctly
    assert len(labels_pred) == len(true_labels)

    # Check performance
    print(metrics.adjusted_mutual_info_score(true_labels, labels_pred))

    # Check if number of clusters is correct
    assert len(ikdc.clusters_) == k

    # Check if number of points in each cluster is correct
    for cluster in ikdc.clusters_:
        assert cluster.n_points > 0

    # Check if the number of iterations is greater than 0
    assert ikdc.n_it > 0


test_IDKC()
