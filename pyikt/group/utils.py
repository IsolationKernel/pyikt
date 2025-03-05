"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from typing import Any, List, Optional, Union

import numpy as np
from sklearn.utils.validation import check_array


def check_format(
    X: Union[List[Any], np.ndarray],
    allow_empty: bool = False,
    n_features: Optional[int] = None,
) -> np.ndarray:
    """
    Validates group data format.

    Parameters:
    -----------
    X : array-like
        Trajectory data with shape (n_trajectories, n_samples, n_features)
    allow_empty : bool, default=False
        Whether to allow trajectories with no samples
    n_features : int, optional, default=None
        Expected number of features

    Returns:
    --------
    np.ndarray
        Validated group data
    """
    # Convert to numpy array if needed
    n_group_features = []

    if not isinstance(X, (list, np.ndarray)):
        raise TypeError("X must be a list or numpy array")
    elif isinstance(X, np.ndarray):
        if X.ndim != 3:
            raise ValueError("X must have shape (n_groups, n_samples, n_features)")
    else:
        for i, group in enumerate(X):
            if not isinstance(group, (list, np.ndarray)):
                raise ValueError(
                    f"Group at index {i} is not array-like. Expected shape: (n_samples, n_features)"
                )
            try:
                group = check_array(group, ensure_2d=True, allow_nd=False)
            except ValueError as e:
                raise ValueError(
                    f"Group at index {i} is not a valid array-like (n_samples, n_features), got error: {e}"
                ) from e

            n_group_features.append(group.shape[1])

        if n_group_features and len(set(n_group_features)) > 1:
            raise ValueError("All groups must have the same number of features")

        if (
            n_features is not None
            and n_group_features
            and n_group_features[0] != n_features
        ):
            raise ValueError(
                f"Expected {n_features} features but got {n_group_features[0]}"
            )

    return X
