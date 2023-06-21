from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils import rnn
from torch import nn

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    InterpretableMultiHeadAttention,
)


class MultiMix_TFT(TemporalFusionTransformer):
    """
    The MultiMix Temporal Fusion Transformer (MultiMix_TFT) is a class used for multi-task learning with
    mixed frequency data. This class enhances the Temporal Fusion Transformer (TFT), a deep learning
    model designed for interpretable time series prediction, to handle multiple targets at
    different frequencies.

    The class accepts multiple forecast targets and an aggregate mode to specify how the attention mechanism (part of the TFT)
    works. It also contains a 'step' method for training and validation (in line with the original pytorch forecasting
    package rather than pytorch lightning), which calculates the losses and
    performs backpropagation.

    Args:
        mf_target (str, List[str]): Names of the target variables for the model.
        mf_filler (int): Placeholder value for missing entries in the mixed frequency target variable.
        agg_mode (str): Aggregation mode for the attention mechanism, can be one of ['mean', 'sum', 'concat'].
        **kwargs: Additional arguments for the parent TemporalFusionTransformer class.
    """

    def __init__(
        self,
        mf_target: Union[str, List[str]],
        mf_filler: int,
        agg_mode: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.agg_mode = agg_mode
        self.multihead_attn = InterpretableMultiHeadAttentionFlex(
            d_model=self.hparams.hidden_size,
            n_head=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
            agg_mode=self.agg_mode,
        )

        self.mf_target = mf_target
        self.mf_filler = mf_filler

    def calculate_loss(self, task, prediction, target, target1_idx, mask, filler):
        if task == self.target_names[1] and mask.sum() == 0:
            return {
                f"loss_{task}": torch.tensor(0.0),
                f"mae_{task}": torch.tensor(0.0),
                f"rmse_{task}": torch.tensor(0.0),
            }

        pred = (
            prediction[target1_idx][mask]
            if task == self.target_names[1]
            else prediction[0]
        )
        target_subset = (
            target[mask].unsqueeze(-1) if task == self.target_names[1] else target[0]
        )
        return {
            f"loss_{task}": self.loss.metrics[target1_idx](pred, target_subset),
            f"mae_{task}": MAE()(pred, target_subset),
            f"rmse_{task}": RMSE()(pred, target_subset),
        }

    def correlation(self, x, y):
        return torch.corrcoef(torch.stack((x, y)))[0][1]

    def step(
        self,
        x: Dict[str, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # pad y sequence if different encoder lengths exist
        if (x["decoder_lengths"] < x["decoder_lengths"].max()).any():
            y = (
                (
                    [
                        rnn.pack_padded_sequence(
                            y_part,
                            lengths=x["decoder_lengths"].cpu(),
                            batch_first=True,
                            enforce_sorted=False,
                        )
                        for y_part in y[0]
                    ],
                    y[1],
                )
                if isinstance(y[0], (list, tuple))
                else (
                    rnn.pack_padded_sequence(
                        y[0],
                        lengths=x["decoder_lengths"].cpu(),
                        batch_first=True,
                        enforce_sorted=False,
                    ),
                    y[1],
                )
            )

        if self.training and len(self.hparams.monotone_constaints) > 0:
            raise NotImplementedError(
                "Monotonicity constraints are not (yet) implemented for the Multimix TFT"
            )

        out = self(x, **kwargs)
        prediction = out["prediction"]

        losses = {}

        task0 = self.target_names[0]
        losses.update(
            self.calculate_loss(
                task0, prediction, y[0], 0, y[0][0] != self.mf_filler, self.mf_filler
            )
        )

        task1 = self.target_names[1]
        target1_idx = self.target_names.index(self.mf_target)
        mask = y[0][target1_idx] != self.mf_filler
        losses.update(
            self.calculate_loss(
                task1, prediction, y[0], target1_idx, mask, self.mf_filler
            )
        )

        loss = (
            self.loss.weights[0] * losses[f"loss_{task0}"]
            + self.loss.weights[1] * losses[f"loss_{task1}"]
        )

        n_samples = len(x["decoder_target"])

        if self.current_stage == "val":
            corr0 = self.correlation(prediction[0].squeeze(), y[0][0].squeeze())
            corr1 = (
                self.correlation(
                    prediction[target1_idx][mask].squeeze(),
                    y[0][target1_idx][mask].unsqueeze(-1).squeeze(),
                )
                if mask.sum() != 0
                else torch.tensor(0.0)
            )
            total_corr = corr0 + corr1
            tot_mae = losses[f"mae_{task0}"] + losses[f"mae_{task1}"]

            self.log(
                f"{self.current_stage}_corr0",
                corr0,
                on_epoch=True,
                batch_size=n_samples,
            )
            self.log(
                f"{self.current_stage}_corr1",
                corr1,
                on_epoch=True,
                batch_size=n_samples,
            )
            self.log(
                f"{self.current_stage}_corr_tot",
                total_corr,
                on_epoch=True,
                batch_size=n_samples,
            )
            self.log(
                f"{self.current_stage}_tot_mae",
                tot_mae,
                on_epoch=True,
                batch_size=n_samples,
            )

        with torch.no_grad():
            y[0][target1_idx] *= mask

        self.log(
            f"{self.current_stage}_loss",
            loss,
            on_step=self.training,
            on_epoch=True,
            prog_bar=True,
            batch_size=n_samples,
        )
        for i, j in losses.items():
            self.log(
                f"{self.current_stage}_{i}",
                j,
                on_step=self.training,
                on_epoch=True,
                batch_size=n_samples,
            )

        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out


class InterpretableMultiHeadAttentionFlex(InterpretableMultiHeadAttention):
    """
    The InterpretableMultiHeadAttentionFlex is an extension of the InterpretableMultiHeadAttention class,
    which is a core component of the Temporal Fusion Transformer.

    This class includes a flexible aggregation mode ('mean', 'sum', or 'concat'), allowing to control
    how information is aggregated across multiple attention heads.

    The forward method of this class applies attention to the inputs and returns the aggregated result.

    Args:
        agg_mode (str): The aggregation mode for attention mechanism, can be one of ['mean', 'sum', 'concat'].
        **kwargs: Additional arguments for the parent InterpretableMultiHeadAttention class.
    """

    def __init__(self, agg_mode: str = "mean", **kwargs):
        self.agg_mode = agg_mode
        super().__init__(**kwargs)

        assert self.agg_mode in [
            "mean",
            "sum",
            "concat",
        ], 'agg_mode not recognized. Should be one of ["mean", "sum", "concat"]'
        if self.agg_mode == "concat":
            # if concatenation, change input dim
            self.w_h = nn.Linear(self.n_head * self.d_v, self.d_model, bias=False)
        else:
            # else, apply normal strategy
            self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        if self.agg_mode == "mean":
            outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        elif self.agg_mode == "sum":
            outputs = torch.sum(head, dim=2) if self.n_head > 1 else head
        elif self.agg_mode == "concat":
            outputs = torch.cat(heads, dim=2)

        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
