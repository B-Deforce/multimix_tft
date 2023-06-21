from pytorch_forecasting.metrics import MAE, RMSE, MultiLoss

from pytorch_optimizer import Ranger21
import pytorch_lightning as pl
import torch
import torch.nn as nn


class MultiMix_LSTM(pl.LightningModule):

    """
    A Multi-Task Learning (MTL) LSTM model for handling Mixed Frequency (MF) data implemented with PyTorch Lightning.

    This model utilizes a Long Short-Term Memory (LSTM) architecture for multi-task learning scenarios where the tasks
    are performed at different frequencies. This makes it suitable for datasets where different features have different
    update frequencies.

    Attributes:
        mf_target (str): The mixed-frequency target for multi-task learning.
        mf_filler (float): The filler value used when a lower frequency target doesn't have a corresponding value
                           in a higher frequency dataset.
        target (list): The list of targets for the LSTM.
        n_features (int): The number of features in the input data.
        learning_rate (float): The learning rate for the optimizer.
        n_layers (int): The number of LSTM layers.
        hidden_size (int): The number of neurons in the hidden layers.
        dropout (float): The dropout rate for regularization.
        output_size (int): The size of the output.
        optimizer (str): The optimizer type.
        mtl_loss (MultiLoss): The loss function for multi-task learning
        reduce_on_plateau_patience (int): The patience parameter for the learning rate scheduler.
    """

    def __init__(
        self,
        mf_target: str,
        mf_filler: float,
        target: list,
        n_features: int,
        learning_rate: float,
        n_layers: int,
        hidden_size: int,
        dropout: float,
        output_size: int,
        optimizer: str,
        mtl_loss: MultiLoss,
        reduce_on_plateau_patience: int = 20,
    ):
        self.loss = MultiMix_Loss(
            self.hparams.target,
            self.hparams.mf_target,
            self.hparams.mf_filler,
            self.hparams.mtl_loss,
        )
        self.lstm = nn.LSTM(
            input_size=self.hparams.n_features,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hparams.hidden_size, out)
                for out in self.hparams.output_size
            ]
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(
            x["encoder_cont"]
        )  # assumes a dataloader obtained through TimeSeriesDataset from Pytorch Forecasting
        last_out = lstm_out[:, -1, :]
        out = [linear(last_out) for linear in self.linears]
        return out

    def shared_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        logs = {
            "loss": loss["loss_tot"],
            "mae_swp": loss["mae_swp_mpa_scaled"],
            "mae_soil": loss["mae_avg_moist_scaled"],
            "tot_mae": loss["tot_mae"],
        }
        return {"loss": loss["loss_tot"], "logs": logs}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def configure_optimizers(self):
        OPTIMIZERS = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "ranger": Ranger21,
        }

        optimizer = OPTIMIZERS[self.hparams.optimizer](
            self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.hparams.reduce_on_plateau_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class MultiMix_Loss:
    """
    A class representing a Multi-Task Learning (MTL) loss function for handling Mixed Frequency (MF) data.

    The MultiMix loss function is responsible for calculating the loss for multiple tasks, rather than a single task.
    This class is specifically designed to handle cases where the tasks operate at different frequencies.

    Attributes:
        target_names (list): The names of the target variables.
        mf_target (str): The mixed frequency target.
        mf_filler (float): The filler value used when the lower frequency target doesn't have a corresponding value
                           in a higher frequency dataset.
        mtl_loss (MultiLoss): The loss function for multi-task learning.
    """

    def __init__(self, target_names, mf_target, mf_filler, mtl_loss):
        self.target_names = target_names
        self.mf_target = mf_target
        self.mf_filler = mf_filler
        self.mtl_loss = mtl_loss

    def calculate_loss(self, task, pred, target, target_index):
        losses = {}
        losses[f"loss_{task}"] = self.mtl_loss.metrics[target_index](pred, target)
        losses[f"mae_{task}"] = MAE()(pred, target)
        losses[f"rmse_{task}"] = RMSE()(pred, target)
        return losses

    def __call__(self, out, y):
        pred = out
        y = y
        losses = {}

        for i, task in enumerate(self.target_names):
            target = y[0][i]
            task_pred = pred[i]

            if (
                task == self.target_names[1]
            ):  # if task is the second one, apply the mask
                mask = target != self.mf_filler
                if mask.sum() > 0:  # there are valid entries
                    target = target[mask].unsqueeze(-1)
                    task_pred = task_pred[mask]
                else:
                    target = torch.tensor(0.0)
                    task_pred = torch.tensor(0.0)

            task_losses = self.calculate_loss(task, task_pred, target, i)
            losses.update(task_losses)

        losses["tot_mae"] = sum(losses[f"mae_{task}"] for task in self.target_names)

        losses["loss_tot"] = sum(
            self.mtl_loss.weights[i] * losses[f"loss_{task}"]
            for i, task in enumerate(self.target_names)
        )
        return losses
