"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from typing import Any, List, Optional, Union

import numpy as np


def check_format(
    X: Union[List[Any], np.ndarray],
    allow_empty: bool = False,
    n_features: Optional[int] = 2,
) -> np.ndarray:
    """
    Validates trajectory data format.

    Parameters:
    -----------
    X : array-like
        Trajectory data with shape (n_trajectories, n_samples, n_features)
    allow_empty : bool, default=False
        Whether to allow trajectories with no samples
    n_features : int, optional, default=2
        Expected number of features (e.g., 2 for longitude/latitude)

    Returns:
    --------
    np.ndarray
        Validated trajectory data
    """
    # Convert to numpy array if needed
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError(
            "X should be array-like with 3 dimensions (n_trajectories, n_samples, n_features)"
        )

    X = np.asarray(X)

    # Validate dimensions
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")

    # Check for empty trajectories
    if not allow_empty:
        empty_mask = np.array([len(traj) == 0 for traj in X])
        if np.any(empty_mask):
            empty_indices = np.where(empty_mask)[0]
            raise ValueError(
                f"Trajectories at indices {empty_indices.tolist()} have no samples"
            )

    # Extract only non-empty trajectories for further checks
    non_empty_mask = np.array([len(traj) > 0 for traj in X])
    if not np.any(non_empty_mask):
        # All trajectories are empty but allowed
        return X

    non_empty_trajectories = X[non_empty_mask]

    # Check trajectory lengths
    trajectory_lengths = np.array([len(traj) for traj in non_empty_trajectories])
    unique_lengths = np.unique(trajectory_lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            f"All trajectories must have same length. Found lengths: {sorted(unique_lengths)}"
        )

    # Check feature dimensions
    if n_features is not None:
        feature_counts = np.array([traj.shape[1] for traj in non_empty_trajectories])
        if len(np.unique(feature_counts)) > 1 or feature_counts[0] != n_features:
            feature_desc = "longitude, latitude" if n_features == 2 else f"{n_features}"
            raise ValueError(
                f"All trajectories must have {n_features} features ({feature_desc})"
            )

    return X
