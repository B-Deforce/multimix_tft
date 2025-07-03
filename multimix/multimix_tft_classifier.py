from typing import Optional

import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryRecall

from multimix.multimix_tft import MultiMixTFT


class MultiMixTFTC(MultiMixTFT):
    """MultiMixTFT model for multi-task time series classification with mixed-frequency time-series.
    This model extends the MultiMixTFT to handle a binary classification task in addition to a regression task.
    Args:
        weights (Optional[torch.Tensor]): Optional weights for the positive class in the binary classification task
            This can be used to handle class imbalance by assigning higher weights to the positive class.
        **kwargs: Additional keyword arguments for the MultiMixTFT.
    """

    def __init__(self, weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # network parameters
        self.reg_loss = nn.MSELoss()
        self.c_loss = (
            nn.BCEWithLogitsLoss(pos_weight=weights)
            if weights is not None
            else nn.BCEWithLogitsLoss()
        )
        self.c_loss_val = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.train_ap = BinaryRecall()
        self.val_ap = BinaryRecall()

        # initialize network components
        self.build_output_feed_forwards()

        # Initializing remaining weights
        self.init_weights()

    def training_step(self, batch, batch_idx):
        """Training step for the MultiMixTFT classification model.
        Args:
            batch: A batch of data containing input features and targets.
            batch_idx: Index of the current batch.
        Returns:
            loss: Computed loss for the batch.
        """
        x, y = batch
        optimizer = self.optimizers()
        y0_hat, y1_hat = self(x)
        y0, y1 = y[:, 0].squeeze(), y[:, 1].squeeze()

        # Loss for the first task
        optimizer.zero_grad()
        loss0 = self.reg_loss(y0_hat, y0)

        # Check for availability of the mixed-frequency target in the batch
        # Create a mask for available targets
        available_mask = ~torch.isnan(y1)

        if available_mask.any():
            y1 = y1[available_mask]
            y1_hat = y1_hat[available_mask]

            loss1 = self.c_loss(y1_hat, y1)  # Compute loss only for available targets
        else:
            loss1 = torch.tensor(0.0).to(self.device)

        loss = self.alpha * loss0 + (1 - self.alpha) * loss1
        self.manual_backward(loss)
        optimizer.step()
        with torch.no_grad():
            if available_mask.any():
                probs = torch.sigmoid(y1_hat).float()
                class_accuracy = self.train_acc(probs, y1)
                class_ap = self.train_ap(probs, y1.int())
            else:
                class_accuracy = torch.tensor(0.0).to(self.device)
                class_ap = torch.tensor(0.0).to(self.device)
            self.log(
                "train_accuracy",
                class_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log("train_ap", class_ap, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_loss0", loss0, prog_bar=True)
            self.log("train_loss1", loss1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the MultiMixTFT classification model.
        Args:
            batch: A batch of data containing input features and targets.
            batch_idx: Index of the current batch.
        Returns:
            loss: Computed loss for the batch.
        """
        x, y = batch
        y0_hat, y1_hat = self(x)
        y0, y1 = y[:, 0].squeeze(), y[:, 1].squeeze()

        # Loss for the first, normal-frequency task
        loss0 = self.reg_loss(y0_hat, y0)

        # Check for availability of the mixed-frequency target in the batch
        # Create a mask for available targets
        available_mask = ~torch.isnan(y1)

        if available_mask.any():
            y1 = y1[available_mask]
            y1_hat = y1_hat[available_mask]

            loss1 = self.c_loss_val(y1_hat, y1)
            probs = torch.sigmoid(y1_hat).float()
            class_accuracy = self.val_acc(probs, y1)
            class_ap = self.val_ap(probs, y1.int())
        else:
            loss1 = torch.tensor(0.0).to(self.device)
            class_accuracy = torch.tensor(0.0).to(self.device)
            class_ap = torch.tensor(0.0).to(self.device)

        # log metrics
        loss = loss0 + loss1
        self.log(
            "val_accuracy", class_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("val_ap", class_ap, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss0", loss0, prog_bar=True)
        self.log("val_loss1", loss1, prog_bar=True)
        return loss

    def build_output_feed_forwards(self):
        """Builds the output feed-forward layers for the MultiMixTFT model."""
        self.output_feed_forward0 = torch.nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size
        )
        self.output_feed_forward1 = torch.nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size
        )

        self.final0 = torch.nn.Linear(self.hidden_layer_size, self.output_size)
        self.final1 = torch.nn.Linear(
            self.hidden_layer_size, 1
        )  # output size is 1 for binary classification
