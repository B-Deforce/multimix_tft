import torch
from torch import nn

from losses import QuantileLoss
from tft.tft import TemporalFusionTransformer


class MultiMixTFT(TemporalFusionTransformer):
    """MultiMixTFT model for multi-task time series forecasting with mixed-frequency time-series.
    This model extends the Temporal Fusion Transformer to handle two tasks
    by predicting two outputs simultaneously. It uses a quantile loss function for
    quantile regression tasks.

    The first task is a normal-frequency task, while the second task is a mixed-frequency task.
    Args:
        window_size (int): Size of the input window.
        quantiles (list[float], optional): List of quantiles for quantile regression. Defaults to None.
        alpha (float, optional): Weighting factor for the two tasks in the loss function. Defaults to 0.5.
        **kwargs: Additional keyword arguments for the TemporalFusionTransformer.
    """
    def __init__(
        self,
        window_size: int,
        quantiles: list[float] = None,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # network parameters
        self.window_size = window_size
        if quantiles:
            self.quantiles = quantiles
            self.loss = QuantileLoss(self.quantiles)
        self.alpha = alpha
        self.automatic_optimization = False

        # initialize network components
        self.build_output_feed_forwards()

        # Initializing remaining weights
        self.init_weights()

    def forward(self, x):
        """ Forward pass of the MultiMixTFT model.
        Args:
            x: Input tensor containing static, historical, and known inputs.
            return_importance: If True, returns feature importance along with outputs.
        Returns:
            output0: Output tensor for the first task.
            output1: Output tensor for the second task.
        """
        static_embedded, historical_embedded, known_embedded = self.get_encoded_inputs(
            x
        )
        if len(static_embedded.shape) == 2:
            static_embedded = static_embedded.unsqueeze(1)
        if len(known_embedded.shape) == 2:
            known_embedded = known_embedded.unsqueeze(1)
        static_encoder, static_importance = self.static_vsn(
            static_embedded
        )  # first categorical, then real (same for historical and known)
        static_context_variable_selection = self.static_context_variable_selection_grn(
            static_encoder
        )
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        historical_features, historical_importance = self.temporal_historical_vsn(
            (historical_embedded, static_context_variable_selection)
        )
        future_features, future_importance = self.temporal_future_vsn(
            (known_embedded, static_context_variable_selection)
        )

        history_lstm, (state_h, state_c) = self.historical_lstm(
            historical_features,
            (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)),
        )

        future_lstm, _ = self.future_lstm(future_features, (state_h, state_c))
        # Apply gated skip connection
        input_embeddings = torch.cat((historical_features, future_features), axis=1)
        lstm_layer = torch.cat((history_lstm, future_lstm), axis=1)
        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(
            lstm_layer, input_embeddings
        )
        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)

        enriched = self.static_enrichment(
            (temporal_feature_layer, expanded_static_context)
        )
        # attention
        x, self_att = self.self_attn_layer(
            enriched, enriched, enriched, mask=self.get_decoder_mask(enriched)
        )
        x = self.post_attn_gate_add_norm(x, enriched)
        decoder = self.GRN_positionwise(x)
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)
        output0 = self.output_feed_forward0(
            transformer_layer[Ellipsis, self.window_size :, :]
        )
        output1 = self.output_feed_forward1(
            transformer_layer[Ellipsis, self.window_size :, :]
        )
        output0 = self.final0(output0).squeeze()
        output1 = self.final1(output1).squeeze()
        return output0, output1

    def training_step(self, batch, batch_idx):
        """Training step for the MultiMixTFT model.
        Args:
            batch: A batch of data containing input features and targets.
            batch_idx: Index of the current batch.
        Returns:
            loss: Computed loss for the batch.
        """
        x, y = batch
        optimizer = self.optimizers() # Get the optimizer, pl built-in
        y0_hat, y1_hat = self(x)
        y0, y1 = y[:, 0].squeeze(), y[:, 1].squeeze()

        # Loss for the first, normal-frequency task
        optimizer.zero_grad()
        loss0 = self.loss(y0_hat, y0)

        # Check for availability of the mixed-frequency target in the batch
        # Create a mask for available targets
        available_mask = ~torch.isnan(y1)  

        if available_mask.any():  
            y1 = y1[available_mask] 
            y1_hat = y1_hat[available_mask]

            loss1 = self.loss(y1_hat, y1)  # Compute loss only for available targets
        else:
            loss1 = torch.tensor(0.0).to(self.device)

        loss = self.alpha * loss0 + (1 - self.alpha) * loss1
        self.manual_backward(loss)  
        optimizer.step()
        with torch.no_grad():
            # get middle idx of quantiles
            if available_mask.any() and self.quantiles is not None:
                middle_idx = len(self.quantiles) // 2
                mse_loss1 = nn.MSELoss()(y1_hat[:, middle_idx], y1)
            elif available_mask.any() and self.quantiles is None:
                mse_loss1 = nn.MSELoss()(y1_hat, y1)
            else:
                mse_loss1 = torch.tensor(0.0).to(self.device)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_loss0", loss0, prog_bar=True)
            self.log("train_loss1", loss1, prog_bar=True)
            self.log("train_mse_loss1", mse_loss1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the MultiMixTFT model.
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
        loss0 = self.loss(y0_hat, y0)
        
        # Check for availability of the mixed-frequency target in the batch
        # Create a mask for available targets
        available_mask = ~torch.isnan(y1)
        
        if available_mask.any():
            y1 = y1[available_mask]
            y1_hat = y1_hat[available_mask]
        
            loss1 = self.loss(y1_hat, y1)
        else:
            loss1 = torch.tensor(0.0).to(self.device)
        
        loss = self.alpha * loss0 + (1 - self.alpha) * loss1
        # log metrics
        if available_mask.any() and self.quantiles is not None:
            middle_idx = len(self.quantiles) // 2
            mse_loss1 = nn.MSELoss()(y1_hat[:, middle_idx], y1)
        elif available_mask.any() and self.quantiles is None:
            mse_loss1 = nn.MSELoss()(y1_hat, y1)
        else:
            mse_loss1 = torch.tensor(0.0).to(self.device)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss0", loss0, prog_bar=True)
        self.log("val_loss1", loss1, prog_bar=True)
        self.log("val_mse_loss1", mse_loss1, prog_bar=True)
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
        self.final1 = torch.nn.Linear(self.hidden_layer_size, self.output_size)