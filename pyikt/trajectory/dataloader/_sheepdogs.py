import datetime

import pandas as pd

from .base import FileDataset


class SheepDogs(FileDataset):
    """SheepDogs trajectory dataset.

    A trajectory dataset collected from MoveBank containing movement patterns of sheep dogs
    and other related animals. This dataset can be used for trajectory analysis and anomaly detection.

    The dataset is loaded with 2 features (longitude and latitude), and samples are classified
    into 2 classes (normal and anomalous).

    Attributes
    ----------
    n_features : int
        Number of features in the dataset (2: longitude and latitude).
    n_samples : int
        Total number of trajectory samples after processing.
    n_classes : int
        Number of classes (2: normal and anomalous).
    anomaly_ratio : float
        Ratio of anomalous trajectories to total trajectories.

    References
    ----------
    .. [1] Movebank: https://www.movebank.org/cms/movebank-main

    .. [2] Wang, Y., Wang, Z., Ting, K. M., & Shang, Y. (2024).
       A Principled Distributional Approach to Trajectory Similarity Measurement and
       its Application to Anomaly Detection. Journal of Artificial Intelligence Research, 79, 865-893.

    Examples
    --------
    >>> from pyikt.trajectory.dataloader import SheepDogs
    >>> sheepdogs = SheepDogs()
    >>> X, y = sheepdogs.load(return_X_y=True)
    """

    def __init__(self):
        super().__init__(
            n_features=2,
            n_samples=None,
            n_classes=2,
            filename="./datasets/sheepdogs.zip",
        )
        self.anomaly_ratio = None

    def load(self, return_X_y=False):
        """Load the SheepDogs dataset.

        Parameters
        ----------
        return_X_y : bool, default=False
            If True, returns a tuple (X, y) where X is the data and y is the target.
            If False, returns a dict with keys 'X' and 'y'.

        Returns
        -------
        dict or tuple
            Either (X, y) tuple or {'X': data, 'y': target} dict where data is a list
            of trajectories and target indicates normal (1) or anomalous (0) trajectories.
        """
        # Load and filter data
        data = pd.read_csv(
            self.path,
            usecols=[
                "timestamp",
                "location-long",
                "location-lat",
                "individual-local-identifier",
            ],
            parse_dates=["timestamp"],
        )

        # Select specific individuals for dataset
        individual_local = data["individual-local-identifier"].value_counts().index
        selected_individuals = [
            0,
            1,
            2,
            6,
            14,
        ]  # Individual #14 is considered anomalous
        mask = data["individual-local-identifier"].isin(
            individual_local[selected_individuals]
        )
        sub_data = data[mask]
        sub_data.dropna(
            subset=["location-long", "location-lat"], how="any", inplace=True
        )

        # Define trajectory splitting parameters
        gap = datetime.timedelta(hours=1)
        tmp_traj = []
        anomalies = []
        normal_traj = []
        labels = []

        # Process trajectories
        for i in range(1, len(sub_data)):
            previous_location = [
                sub_data["location-long"].iloc[i - 1],
                sub_data["location-lat"].iloc[i - 1],
            ]
            previous_timestamp = sub_data["timestamp"].iloc[i - 1]
            current_timestamp = sub_data["timestamp"].iloc[i]
            previous_individual = sub_data["individual-local-identifier"].iloc[i - 1]
            current_individual = sub_data["individual-local-identifier"].iloc[i]
            tmp_traj.append(previous_location)

            # Split trajectory when there's a large time gap or different individual
            if (
                current_timestamp - previous_timestamp > gap
                or previous_individual != current_individual
            ):
                if len(tmp_traj) > 10:  # Ensure minimum trajectory length
                    if previous_individual == individual_local[14]:
                        anomalies.append(tmp_traj)
                        labels.append(0)  # 0 for anomalous
                    else:
                        normal_traj.append(tmp_traj)
                        labels.append(1)  # 1 for normal
                tmp_traj = []

        # Filter anomalies with a minimum length requirement
        filtered_anomalies = [x for x in anomalies if len(x) > 16]
        all_traj = normal_traj + filtered_anomalies

        # Update object attributes
        self.anomaly_ratio = len(filtered_anomalies) / len(all_traj)
        self.n_samples = len(all_traj)

        if return_X_y:
            return all_traj, labels
        else:
            return {
                "X": all_traj,
                "y": labels,
            }
