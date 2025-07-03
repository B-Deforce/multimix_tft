import torch
import torch.nn as nn
import pytorch_lightning as pl


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        """
        Initializes the QuantileLoss module.

        Parameters:
        - quantiles: A list of quantiles, each a float between 0 and 1.
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_preds, y_true):
        """
        Computes the quantile loss for multiple quantiles.

        Parameters:
        - y_true: The true target values, a tensor of shape (batch_size,).
        - y_preds: The predicted values, a tensor of shape (batch_size, num_quantiles),
                   where num_quantiles is the number of quantiles for which predictions were made.

        Returns:
        - loss: The average quantile loss across all quantiles.
        """
        assert (
            len(self.quantiles) == y_preds.shape[1]
        ), "Number of predictions must match number of quantiles"

        errors = (
            y_true.unsqueeze(1) - y_preds
        )  # Broadcast true values across quantile dimension
        losses = torch.zeros_like(errors)

        for i, tau in enumerate(self.quantiles):
            losses[:, i] = torch.where(
                errors[:, i] > 0, tau * errors[:, i], (tau - 1) * errors[:, i]
            )

        return 2 * losses.mean()


class CustomLoss(pl.LightningModule):
    def __init__(self, l_limit, u_limit, penalty):
        super().__init__()
        print("Using custom loss function")
        self.l_limit = l_limit
        self.u_limit = u_limit
        self.penalty = penalty

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 1]
        error = y_true - y_pred
        condition = torch.logical_and(y_true <= self.u_limit, y_true >= self.l_limit)
        heavy_penalty = torch.where(condition, error**2 * self.penalty, error**2)
        return torch.mean(heavy_penalty)
