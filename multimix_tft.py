import pytorch_lightning as pl
from tft.base_components import GatedLinearUnit
from tft.base_components import GateAddNormNetwork
from tft.base_components import GatedResidualNetwork
from tft.base_components import ScaledDotProductAttention
from tft.base_components import InterpretableMultiHeadAttention
from tft.base_components import VariableSelectionNetwork
import torch
from torch import nn
from typing import List, Dict, Tuple, Optional
from losses import QuantileLoss


class TemporalFusionTransformer(pl.LightningModule):
    def __init__(
        self,
        hidden_layer_size,
        static_categorical_sizes: Dict[str, int],
        historical_categorical_sizes: Dict[str, int],
        static_reals: List[str],
        historical_reals: List[str],
        known_categoricals: List[str],
        known_reals: List[str],
        dropout_rate: float,
        num_heads: int,
        output_size: int,
        window_size: int,
        quantiles: List[float] = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # network parameters
        self.hidden_layer_size = hidden_layer_size
        self.static_categorical_sizes = static_categorical_sizes
        self.historical_categorical_sizes = historical_categorical_sizes
        self.static_reals = static_reals
        self.historical_reals = historical_reals
        self.known_categoricals = known_categoricals
        self.known_reals = known_reals
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.output_size = output_size
        self.window_size = window_size
        self.quantiles = quantiles
        self.loss = QuantileLoss(self.quantiles)
        # self.loss = nn.L1Loss()
        self.automatic_optimization = False
        self.lr = lr

        self.static_cat_length = (
            len(self.static_categorical_sizes.keys())
            if self.static_categorical_sizes
            else 0
        )
        self.static_real_length = len(self.static_reals) if self.static_reals else 0
        self.historical_cat_length = (
            len(self.historical_categorical_sizes.keys())
            if self.historical_categorical_sizes
            else 0
        )
        self.historical_real_length = (
            len(self.historical_reals) if self.historical_reals else 0
        )
        self.known_cat_length = (
            len(self.known_categoricals) if self.known_categoricals else 0
        )
        self.known_real_length = len(self.known_reals) if self.known_reals else 0

        # initialize network components
        if self.static_categorical_sizes:
            self.static_cat_embedding = nn.ModuleDict(
                {
                    var_name: nn.Embedding(cardinality, self.hidden_layer_size)
                    for var_name, cardinality in self.static_categorical_sizes.items()
                }
            )
        if self.static_reals:
            self.static_real_embedding = nn.ModuleDict(
                {
                    var_name: nn.Linear(1, self.hidden_layer_size)
                    for var_name in self.static_reals
                }
            )
        if self.historical_categorical_sizes:
            self.historical_cat_embedding = nn.ModuleDict(
                {
                    var_name: nn.Embedding(cardinality, self.hidden_layer_size)
                    for var_name, cardinality in self.historical_categorical_sizes.items()
                }
            )
        if self.historical_reals:
            self.historical_real_embedding = nn.ModuleDict(
                {
                    var_name: nn.Linear(1, self.hidden_layer_size)
                    for var_name in self.historical_reals
                }
            )
        self.build_variable_selection_networks()
        self.build_static_context_networks()
        self.build_lstm()
        self.build_post_lstm_gate_add_norm()
        self.build_static_enrichment()
        self.build_temporal_self_attention()
        self.build_position_wise_feed_forward()
        self.build_output_feed_forwards()
        self.final0 = nn.Linear(self.hidden_layer_size, self.output_size)
        self.final1 = nn.Linear(self.hidden_layer_size, self.output_size)
        ## Initializing remaining weights
        self.init_weights()

    def forward(self, x):
        static_embedded, historical_embedded, known_embedded = self.get_encoded_inputs(
            x
        )
        if len(static_embedded.shape) == 2:
            static_embedded = static_embedded.unsqueeze(1)
        if len(known_embedded.shape) == 2:
            known_embedded = known_embedded.unsqueeze(1)
        static_encoder, _ = self.static_vsn(static_embedded)
        static_context_variable_selection = self.static_context_variable_selection_grn(
            static_encoder
        )
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        historical_features, _ = self.temporal_historical_vsn(
            (historical_embedded, static_context_variable_selection)
        )
        future_features, _ = self.temporal_future_vsn(
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
        output0 = self.final0(output0)
        output1 = self.final1(output1)
        return output0, output1

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer = self.optimizers()
        y0_hat, y1_hat = self(x)
        y0_hat, y1_hat = y0_hat.squeeze(), y1_hat.squeeze()
        y0, y1 = y[:, 0].squeeze(), y[:, 1].squeeze()

        # Loss for the first task
        optimizer.zero_grad()
        loss0 = self.loss(y0_hat, y0)
        # self.manual_backward(loss0, retain_graph=True)  # Compute gradients for task 1

        # Check for availability of the second target in the batch
        available_mask = ~torch.isnan(y1)  # Create a mask for available targets

        if available_mask.any():  # If there's at least one available target
            y1 = y1[available_mask]  # Filter using the mask
            y1_hat = y1_hat[available_mask]  # Filter the predictions as well

            loss1 = self.loss(y1_hat, y1)  # Compute loss only for available targets
            # self.manual_backward(
            #    loss1, retain_graph=True
            # )  # Compute gradients for task 2
        else:
            loss1 = torch.tensor(0.0).to(self.device)

        self.manual_backward(loss0 + loss1)  # Compute gradients for both tasks
        optimizer.step()
        with torch.no_grad():
            total_loss = loss0 + loss1
            # get middle idx of quantiles
            if self.quantiles is not None:
                middle_idx = len(self.quantiles) // 2
                mse_loss1 = nn.MSELoss()(y1_hat[:, middle_idx], y1)
            else:
                mse_loss1 = nn.MSELoss()(y1_hat, y1)
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("train_loss0", loss0, prog_bar=True)
            self.log("train_loss1", loss1, prog_bar=True)
            self.log("train_mse_loss1", mse_loss1, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y0_hat, y1_hat = self(x)
        y0, y1 = y[:, 0].squeeze(), y[:, 1].squeeze()
        y0_hat, y1_hat = y0_hat.squeeze(), y1_hat.squeeze()
        loss0 = self.loss(y0_hat, y0)
        available_mask = ~torch.isnan(y1)
        if available_mask.any():
            y1 = y1[available_mask]
            y1_hat = y1_hat[available_mask]
            loss1 = self.loss(y1_hat, y1)
            if self.quantiles is not None:
                middle_idx = len(self.quantiles) // 2
                mse_loss1 = nn.MSELoss()(y1_hat[:, middle_idx], y1)
            else:
                mse_loss1 = nn.MSELoss()(y1_hat, y1)
        else:
            loss1 = torch.tensor(0.0).to(self.device)
            mse_loss1 = torch.tensor(0.0).to(self.device)
        loss = loss0 + loss1
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss0", loss0, prog_bar=True)
        self.log("val_loss1", loss1, prog_bar=True)
        self.log("val_mse_loss1", mse_loss1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def init_weights(self):
        for name, p in self.named_parameters():
            if ("lstm" in name and "ih" in name) and "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            elif ("lstm" in name and "hh" in name) and "bias" not in name:
                torch.nn.init.orthogonal_(p)
            elif "lstm" in name and "bias" in name:
                torch.nn.init.zeros_(p)

    def get_encoded_inputs(self, x):
        static_real_input = x.get("static_real", None)
        static_cat_input = x.get("static_cat", None)
        historical_cat_input = x.get("historical_cat", None)
        historical_real_input = x.get("historical_real", None)
        known_cat_input = x.get("known_cat", None)
        known_real_input = x.get("known_real", None)

        # embed static categorical variables
        static_embedded = []
        if static_cat_input is not None:
            static_embedded += [
                self.static_cat_embedding[key](static_cat_input[key][:, 0])
                for key in self.static_cat_embedding.keys()
            ]
        # embed static real variables
        if static_real_input is not None:
            static_embedded += [
                self.static_real_embedding[key](static_real_input[key][:, 0])
                for key in self.static_real_embedding.keys()
            ]

        # concat embedded static and real variables if both are present
        if static_embedded != []:
            static_embedded = torch.cat(static_embedded, dim=1)

        # embed historical cat variables
        historical_embedded = []
        if historical_cat_input is not None:
            historical_embedded += [
                self.historical_cat_embedding[key](historical_cat_input[key])
                for key in self.historical_cat_embedding.keys()
            ]
        # embed historical real variables
        if historical_real_input is not None:
            historical_embedded += [
                self.historical_real_embedding[key](
                    historical_real_input[key].unsqueeze(2)
                )
                for key in self.historical_real_embedding.keys()
            ]

        # concat embedded historical cat and real variables if both are present
        if historical_embedded != []:
            historical_embedded = torch.cat(historical_embedded, dim=2)

        # embed known cat variables
        known_embedded = []
        if known_cat_input is not None:
            assert all(
                [
                    key in self.historical_cat_embedding.keys()
                    for key in known_cat_input.keys()
                ]
            ), "known_cat_input contains keys not in historical_cat_embedding"
            known_embedded += [
                self.historical_cat_embedding[key](known_cat_input[key])
                for key in known_cat_input.keys()
            ]
        # embed known real variables
        if known_real_input is not None:
            assert all(
                [
                    key in self.historical_real_embedding.keys()
                    for key in known_real_input.keys()
                ]
            ), "known_real_input contains keys not in historical_real_embedding"
            known_embedded += [
                self.historical_real_embedding[key](known_real_input[key].unsqueeze(1))
                for key in known_real_input.keys()
            ]

        # concat embedded known cat and real variables if both are present
        if known_embedded != []:
            known_embedded = torch.cat(known_embedded, dim=1)

        return static_embedded, historical_embedded, known_embedded

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs, 1, 1).to(torch.float32)

        return mask.to(self.device)

    def build_variable_selection_networks(self):
        self.static_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size
            * (self.static_cat_length + self.static_real_length),
            output_size=self.static_cat_length + self.static_real_length,
            dropout_rate=self.dropout_rate,
        )

        self.temporal_historical_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size
            * (self.historical_cat_length + self.historical_real_length),
            output_size=self.historical_cat_length + self.historical_real_length,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

        self.temporal_future_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size
            * (self.known_cat_length + self.known_real_length),
            output_size=self.known_cat_length + self.known_real_length,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

    def build_static_context_networks(self):
        self.static_context_variable_selection_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_enrichment_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_state_h_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_state_c_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

    def build_lstm(self):
        self.historical_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
        )
        self.future_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
        )

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

    def build_temporal_self_attention(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_layer_size,
            dropout=self.dropout_rate,
        )

        self.post_attn_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.post_tfd_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_output_feed_forwards(self):
        self.output_feed_forward0 = torch.nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size
        )
        self.output_feed_forward1 = torch.nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size
        )
