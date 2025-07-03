import pandas as pd
import pytorch_lightning as pl
import torch
import typing as tp
from torch.utils.data import DataLoader, Dataset


class TFTDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        static_real_cols: list[str],
        static_cat_cols: list[str],
        historical_real_cols: list[str],
        historical_cat_cols: list[str],
        known_real_cols: list[str],
        known_cat_cols: list[str],
        target: list[str],
        window_size: int,
        group_ids: str,
        time_idx: str,
        time_gap=1,  # time_gap=1 for 1-step prediction, time_gap=2 for 2-step ahead prediction, etc.
    ):
        """Initialize the TFTDataset.
        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            static_real_cols (list[str]): List of static real-valued column names.
            static_cat_cols (list[str]): List of static categorical column names.
            historical_real_cols (list[str]): List of historical real-valued column names.
            historical_cat_cols (list[str]): List of historical categorical column names.
            known_real_cols (list[str]): List of future known real-valued column names.
            known_cat_cols (list[str]): List of future known categorical column names.
            target (list[str]): List of target variable names, as it is multi-task.
            window_size (int): Size of the input window for the model.
            group_ids (str): Column name for group IDs in the dataset.
            time_idx (str): Column name for time index in the dataset.
            time_gap (int): Time gap for the mixed-frequency target, default is 1. Note that this is
                expressed as the number of time steps, not the actual time unit (e.g., hours, days).
                For example, if the time gap is 2, and the time-unit is 4 hours,
                then the model will predict the mixed-frequency target 8 hours ahead.
        """
        super().__init__()
        self.static_real_cols = static_real_cols
        self.static_cat_cols = static_cat_cols
        self.historical_real_cols = historical_real_cols
        self.historical_cat_cols = historical_cat_cols
        self.known_real_cols = known_real_cols
        self.known_cat_cols = known_cat_cols
        self.time_gap = time_gap
        if self.static_real_cols is not None:
            self.static_real_data = torch.tensor(
                data[static_real_cols].values, dtype=torch.float
            )
        if self.static_cat_cols is not None:
            self.static_cat_data = torch.tensor(
                data[static_cat_cols].values, dtype=torch.long
            )
        if self.historical_real_cols is not None:
            self.historical_real_data = torch.tensor(
                data[historical_real_cols].values, dtype=torch.float
            )
        if self.historical_cat_cols is not None:
            self.historical_cat_data = torch.tensor(
                data[historical_cat_cols].values, dtype=torch.long
            )
        if self.known_real_cols is not None:
            self.known_real_data = torch.tensor(
                data[known_real_cols].values, dtype=torch.float
            )
        if self.known_cat_cols is not None:
            self.known_cat_data = torch.tensor(
                data[known_cat_cols].values, dtype=torch.long
            )
        self.target = torch.tensor(data[target].values, dtype=torch.float)
        self.group_ids = torch.tensor(data[group_ids].values, dtype=torch.float)
        self.time_idx = torch.tensor(data[time_idx].values, dtype=torch.float)
        self.window_size = window_size
        self.indices = self.prepare_indices()

    def prepare_indices(self):
        """Prepare indices for the dataset based on the group IDs and time index.
        This method creates sliding windows of indices for each group in the dataset,
        ensuring that the indices are sorted by time index within each group.
        Returns:
            list: A list of indices, where each index is a list of indices for a sliding window.
        """
        indices = []
        unique_groups = torch.unique(self.group_ids)
        for group in unique_groups:
            group_idx = torch.where(self.group_ids == group)[0]

            # Sort the indices in this group by time_idx
            time_idx_group = self.time_idx[group_idx]
            sorted_indices = group_idx[torch.argsort(time_idx_group)]

            # Create windows from the sorted indices
            for i in range(len(sorted_indices) - self.window_size - self.time_gap):
                indices.append(sorted_indices[i : i + self.window_size + self.time_gap])
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """Get a sample from the dataset at the specified index.
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the sample data and the target value.
        """
        sample = {}
        idx = self.indices[index][: -self.time_gap]
        next_idx = self.indices[index][-1]
        if self.static_real_cols is not None:
            sample["static_real"] = self._get_static_real_sample(idx)
        if self.static_cat_cols is not None:
            sample["static_cat"] = self._get_static_cat_sample(idx)
        if self.historical_real_cols is not None:
            sample["historical_real"] = self._get_historical_real_sample(idx)
        if self.historical_cat_cols is not None:
            sample["historical_cat"] = self._get_historical_cat_sample(idx)
        if self.known_real_cols is not None:
            sample["known_real"] = self._get_known_real_sample(next_idx)
        if self.known_cat_cols is not None:
            sample["known_cat"] = self._get_known_cat_sample(next_idx)
        sample["time_idx"] = idx
        sample["group_ids"] = self.group_ids[idx]
        target = self.target[next_idx]
        assert idx[-1] == next_idx - self.time_gap, (
            "Time gap is not correct! Are you sure the index of your dataset starts from 0 and is continuous?"
        )
        return sample, target

    def _get_static_real_sample(self, idx):
        return {
            name: self.static_real_data[idx, i]
            for i, name in enumerate(self.static_real_cols)
        }

    def _get_static_cat_sample(self, idx):
        return {
            name: self.static_cat_data[idx, i]
            for i, name in enumerate(self.static_cat_cols)
        }

    def _get_historical_real_sample(self, idx):
        return {
            name: self.historical_real_data[idx, i]
            for i, name in enumerate(self.historical_real_cols)
        }

    def _get_historical_cat_sample(self, idx):
        return {
            name: self.historical_cat_data[idx, i]
            for i, name in enumerate(self.historical_cat_cols)
        }

    def _get_known_real_sample(self, future_idx):
        return {
            name: self.known_real_data[future_idx, i]
            for i, name in enumerate(self.known_real_cols)
        }

    def _get_known_cat_sample(self, future_idx):
        return {
            name: self.known_cat_data[future_idx, i]
            for i, name in enumerate(self.known_cat_cols)
        }


class TimeSeriesDataLoader(pl.LightningDataModule):
    """Data module for loading time series data for MultiMix TFT model.
    Args:
        train (pd.DataFrame): Training dataset.
        val (pd.DataFrame): Validation dataset.
        static_real_cols (list[str]): List of static real-valued column names.
        static_cat_cols (list[str]): List of static categorical column names.
        historical_real_cols (list[str]): List of historical real-valued column names.
        historical_cat_cols (list[str]): List of historical categorical column names.
        known_real_cols (list[str]): List of future known real-valued column names.
        known_cat_cols (list[str]): List of future known categorical column names.
        primary_target (str): Primary target column name, at normal frequency.
        mf_target (str): Mixed-frequency target column name.
        window_size (int): Size of the input window for the model.
        group_ids (str): Column name for group IDs in the dataset.
        time_idx (str): Column name for time index in the dataset.
        batch_size (int): Batch size for training and validation.
        classification_target (str | None): Optional classification target column name for computing class weights.
        test (pd.DataFrame | None): Optional test dataset.
        time_gap (int): Time gap for the mixed-frequency target, default is 1. Note that this is
            expressed as the number of time steps, not the actual time unit (e.g., hours, days).
            For example, if the time gap is 2, and the time-unit is 4 hours,
            then the model will predict the mixed-frequency target 8 hours ahead.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        static_real_cols: list[str],
        static_cat_cols: list[str],
        historical_real_cols: list[str],
        historical_cat_cols: list[str],
        known_real_cols: list[str],
        known_cat_cols: list[str],
        primary_target: str,
        mf_target: str,
        window_size: int,
        group_ids: str,
        time_idx: str,
        batch_size: int,
        classification_target: tp.Optional[str] = None,
        test: tp.Optional[pd.DataFrame] = None,
        time_gap: int = 1,
    ):
        super().__init__()
        self.train = train
        if classification_target is not None:
            self.weights = self.compute_class_weights(train[classification_target])[
                1
            ].unsqueeze(0)
        self.val = val
        self.test = test
        self.static_real_cols = static_real_cols
        self.static_cat_cols = static_cat_cols
        self.historical_real_cols = historical_real_cols
        self.historical_cat_cols = historical_cat_cols
        self.known_real_cols = known_real_cols
        self.known_cat_cols = known_cat_cols
        self.target = [
            primary_target,
            mf_target,
        ]  # mf target should be second in the list
        self.static_cat_mapper = self.get_cat_mappings(self.static_cat_cols)
        self.static_cat_sizes = self.get_cat_sizes(self.static_cat_mapper)
        self.historical_cat_mapper = self.get_cat_mappings(self.historical_cat_cols)
        self.historical_cat_sizes = self.get_cat_sizes(self.historical_cat_mapper)
        self.window_size = window_size
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.time_idx = time_idx
        self.time_gap = time_gap
        self.train_dataset = TFTDataset(
            self.train,
            self.static_real_cols,
            self.static_cat_cols,
            self.historical_real_cols,
            self.historical_cat_cols,
            self.known_real_cols,
            self.known_cat_cols,
            self.target,
            self.window_size,
            self.group_ids,
            self.time_idx,
            time_gap=self.time_gap,
        )
        self.val_dataset = TFTDataset(
            self.val,
            self.static_real_cols,
            self.static_cat_cols,
            self.historical_real_cols,
            self.historical_cat_cols,
            self.known_real_cols,
            self.known_cat_cols,
            self.target,
            self.window_size,
            self.group_ids,
            self.time_idx,
            time_gap=self.time_gap,
        )
        if self.test is not None:
            self.test_dataset = TFTDataset(
                self.test,
                self.static_real_cols,
                self.static_cat_cols,
                self.historical_real_cols,
                self.historical_cat_cols,
                self.known_real_cols,
                self.known_cat_cols,
                self.target,
                self.window_size,
                self.group_ids,
                self.time_idx,
                time_gap=self.time_gap,
            )

    def compute_class_weights(self, target: pd.Series) -> torch.Tensor:
        """Calculate class weights for imbalanced classes.
        Args:
            target (pd.Series): The target variable containing class labels.
        Returns:
            torch.Tensor: A tensor containing the class weights.
        """
        class_counts = target.value_counts().sort_index()
        num_classes = len(class_counts)
        total = class_counts.sum()
        weights = total / (num_classes * class_counts)
        weights = weights.sort_index()

        return torch.tensor(weights.values, dtype=torch.float32)

    def train_dataloader(self):
        """Create a DataLoader for the training dataset.
        Returns:
            DataLoader: A DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Create a DataLoader for the validation dataset.
        Returns:
            DataLoader: A DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create a DataLoader for the test dataset.
        Returns:
            DataLoader: A DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
        )

    def get_cat_mappings(self, cat_cols):
        """Get categorical mappings for the specified columns.
        Args:
            cat_cols (list): List of categorical column names.
        Returns:
            dict: A dictionary mapping column names to their categorical mappings.
        """
        cat_mapper = {}
        for col in cat_cols:
            self.train[col], cat_mapper[col] = self._map_colname(self.train, col)
            self.val[col].replace(
                {v: k for k, v in cat_mapper[col].items()}, inplace=True
            )
            self.test[col].replace(
                {v: k for k, v in cat_mapper[col].items()}, inplace=True
            )
        return cat_mapper

    def get_cat_sizes(self, cat_mapper: dict[str, list[str]]) -> dict[str, int]:
        """Get sizes of categorical mappings.
        Args:
            cat_mapper (dict): A dictionary mapping column names to their categorical mappings.
        Returns:
            dict: A dictionary containing the sizes of each categorical mapping.
        """
        cat_sizes = {}
        for k in cat_mapper.keys():
            cat_sizes[k] = len(cat_mapper[k])
        return cat_sizes

    def _map_colname(self, data, colname):
        """Map categorical column names to numerical labels.
        Args:
            data (pd.DataFrame): The DataFrame containing the column to be mapped.
            colname (str): The name of the column to be mapped.
        Returns:
            tuple: A tuple containing the numerical labels and a mapping dictionary.
        """
        labels, unique = pd.factorize(data[colname])
        mapper = {k: v for k, v in enumerate(unique)}
        return labels, mapper
