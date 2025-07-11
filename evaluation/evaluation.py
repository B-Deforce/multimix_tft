import typing as tp

import numpy as np
from beartype import beartype
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader

from evaluation.utils import inverse_minmax
from multimix.multimix_tft import MultiMixTFT
from multimix.multimix_tft_classifier import MultiMixTFTC
from multimix.transfer_multimix_tft import MultiMixTFTT


@beartype
class TrainedModel:
    """A class to handle the loading and evaluation of a trained MultiMixTFT model.
    This class supports different tasks such as regression, classification, and transfer learning.
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        task (str): The task type, which can be one of the following:
            - "regression"
            - "classification"
            - "regression_transfer_partial_ft"
            - "regression_transfer_full_ft"
    """

    def __init__(
        self,
        checkpoint_path: str,
        task: tp.Literal[
            "regression",
            "classification",
            "regression_transfer_partial_ft",
            "regression_transfer_full_ft",
        ],
    ):
        self.checkpoint_path = checkpoint_path
        self.task = task
        if self.task == "regression_transfer_partial_ft":
            self.model = MultiMixTFTT.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
            )
        elif self.task == "classification":
            self.model = MultiMixTFTC.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
            )
        else:
            self.model = MultiMixTFT.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
            )

    def _get_predictions(self, dataloader: DataLoader, return_x: bool = False):
        x_batches = []
        y_batches = []
        dl_iterator = iter(dataloader)
        print("Sampling 500 batches from the dataloader...")
        for i in range(500):
            try:
                x, y = next(dl_iterator)
                x_batches.append(x)
                y_batches.append(y)
            except StopIteration:
                print(f"Dataloader exhausted after {i} batches.")
                break

        y_pred, y_true = [], []
        x_all = {"croptype": [], "treatment": [], "field_id": []}
        print("Making predictions...")
        for i, batch in enumerate(x_batches):
            x = batch
            y = y_batches[i][:, 1].numpy().squeeze()
            y_hat = self.model.eval()(x)[1].detach().numpy()

            y_pred.append(y_hat)
            y_true.append(y)

            static_cat = x.get("static_cat", {})
            for key in x_all:
                if key in static_cat:
                    value = static_cat[key].max(axis=1)[0].numpy()
                    x_all[key].append(value)

        # Only include non-empty keys
        x_all = {k: v for k, v in x_all.items() if v}
        return y_true, y_pred, x_all if return_x else None

    def _clean_predictions(
        self,
        y_true: list[np.ndarray],
        y_pred: list[np.ndarray],
        scaling_params: dict[str, dict[str, float]],
        x_all: dict[str, np.ndarray] | None,
        target_col: str,
    ):
        """Cleans the predictions by removing NaNs and applying inverse scaling.
        Args:
            y_true (np.array): True target values.
            y_pred (np.array): Predicted target values.
            scaling_params (dict): Scaling parameters for inverse scaling.
            x_all (dict): Additional features to return.
            target_col (str): The target column for which scaling is applied.
        """
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        x_all = {key: np.concatenate(value) for key, value in x_all.items()} if x_all else None

        # Remove nans
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        x_all = {key: x[mask] for key, x in x_all.items()} if x_all else None

        # Inverse scaling
        if "min_nans" in scaling_params.keys():
            if target_col in scaling_params["min_nans"].keys():
                print(f"Unscaling {target_col}")
                y_true = inverse_minmax(
                    y_true,
                    scaling_max=scaling_params["max_nans"][target_col],
                    scaling_min=scaling_params["min_nans"][target_col],
                )
                y_pred = inverse_minmax(
                    y_pred,
                    scaling_max=scaling_params["max_nans"][target_col],
                    scaling_min=scaling_params["min_nans"][target_col],
                )
        else:
            input("No scaling parameters found. Press any key to continue.")
        return y_true, y_pred, x_all

    def predict(
        self,
        dataloader: DataLoader,
        scaling_params: dict[str, dict[str, float]],
        target_col: str,
        return_x: bool = False,
    ):
        """Predicts the target values using the trained model.
        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            scaling_params (dict): Scaling parameters for inverse scaling.
            target_col (str): The target column for which predictions are made.
            return_x (bool): Whether to return additional features.
        Returns:
            y_true, y_pred, x_all: True and predicted values, and additional features if requested.
        """
        y_true, y_pred, x_all = self._get_predictions(dataloader, return_x)
        y_true, y_pred, x_all = self._clean_predictions(
            y_true, y_pred, scaling_params, x_all, target_col
        )

        return y_true, y_pred, x_all if return_x else None

    def get_regression_metrics(self, y_true, y_pred):
        """Calculates regression metrics: MAE, correlation, MSE, and RMSE.
        Args:
            y_true (np.array): True target values.
            y_pred (np.array): Predicted target values.
        Returns:
            tuple: MAE, correlation, MSE, RMSE.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return mae, corr, mse, rmse
