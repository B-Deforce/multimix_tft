import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class TFTDataset(Dataset):
    def __init__(
        self,
        data,
        static_real_cols,
        static_cat_cols,
        historical_real_cols,
        historical_cat_cols,
        known_real_cols,
        known_cat_cols,
        target,
        window_size,
        group_ids,
        time_idx,
        mixed_only,
        mixed_idx=None,
        time_gap=1,  # time_gap=1 for 1-step prediction, time_gap=2 for 2-step ahead prediction, etc.
    ):
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
        self.mixed_only = mixed_only
        self.mixed_idx = mixed_idx
        self.indices = self.prepare_indices()

    def prepare_indices(self):
        indices = []
        unique_groups = torch.unique(self.group_ids)
        for group in unique_groups:
            group_idx = torch.where(self.group_ids == group)[0]

            # Sort the indices in this group by time_idx
            time_idx_group = self.time_idx[group_idx]
            sorted_indices = group_idx[torch.argsort(time_idx_group)]

            # Create windows from the sorted indices
            for i in range(len(sorted_indices) - self.window_size - self.time_gap):
                if (self.mixed_only) and (
                    self.target[
                        sorted_indices[i + self.window_size + self.time_gap],
                        self.mixed_idx,
                    ]
                    .isnan()
                    .item()
                    == True
                ):
                    continue
                else:
                    indices.append(
                        sorted_indices[i : i + self.window_size + self.time_gap]
                    )
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = {}
        idx = self.indices[index][: -self.time_gap]
        next_idx = self.indices[index][-1]
        # sample["static_real"] = (
        # self._get_static_real_sample(idx)
        # if self.static_real_cols is not None
        # else None
        # )
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
        assert idx[-1] == next_idx - self.time_gap, "Time gap is not correct!"
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
    def __init__(
        self,
        train,
        val,
        static_real_cols,
        static_cat_cols,
        historical_real_cols,
        historical_cat_cols,
        known_real_cols,
        known_cat_cols,
        target,
        window_size,
        group_ids,
        batch_size,
        time_idx,
        test=None,
        mixed_only=False,
        time_gap=1,
        mixed_idx=None,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.static_real_cols = static_real_cols
        self.static_cat_cols = static_cat_cols
        self.historical_real_cols = historical_real_cols
        self.historical_cat_cols = historical_cat_cols
        self.known_real_cols = known_real_cols
        self.known_cat_cols = known_cat_cols
        self.target = target
        self.static_cat_mapper = self.get_cat_mappings(self.static_cat_cols)
        self.static_cat_sizes = self.get_cat_sizes(self.static_cat_mapper)
        self.historical_cat_mapper = self.get_cat_mappings(self.historical_cat_cols)
        self.historical_cat_sizes = self.get_cat_sizes(self.historical_cat_mapper)
        self.window_size = window_size
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.time_idx = time_idx
        self.mixed_only = mixed_only
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
            self.mixed_only,
            mixed_idx=None if self.mixed_only is False else mixed_idx,
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
            self.mixed_only,
            mixed_idx=None if self.mixed_only is False else mixed_idx,
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
                self.mixed_only,
                mixed_idx=None if self.mixed_only is False else mixed_idx,
                time_gap=self.time_gap,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size * 10, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size * 10, shuffle=False
        )

    def get_cat_mappings(self, cat_cols):
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

    def get_cat_sizes(self, cat_mapper):
        cat_sizes = {}
        for k in cat_mapper.keys():
            cat_sizes[k] = len(cat_mapper[k])
        return cat_sizes

    def _map_colname(self, data, colname):
        labels, unique = pd.factorize(data[colname])
        mapper = {k: v for k, v in enumerate(unique)}
        return labels, mapper
