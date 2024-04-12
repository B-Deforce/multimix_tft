from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import tanh
from torch import nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import sklearn.preprocessing


class GatedLinearUnit(pl.LightningModule):
    def __init__(
        self, input_size, hidden_layer_size, dropout_rate=None, activation=None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = nn.Linear(self.input_size, self.hidden_layer_size)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" not in n:
                nn.init.xavier_uniform_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x):
        if self.dropout_rate:
            x = self.dropout(x)

        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


class GateAddNormNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, activation=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.GLU = GatedLinearUnit(
            self.input_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=self.activation_name,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)

        return output


class GatedResidualNetwork(pl.LightningModule):
    def __init__(
        self,
        hidden_layer_size,
        input_size=None,
        output_size=None,
        dropout_rate=None,
        additional_context=None,
        return_gate=False,
    ):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate

        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.ELU = nn.ELU()

        if self.additional_context:
            self.W3 = torch.nn.Linear(
                self.additional_context, self.hidden_layer_size, bias=False
            )

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(
                self.hidden_layer_size, self.output_size, self.dropout_rate
            )
        else:
            self.glu_add_norm = GateAddNormNetwork(
                self.hidden_layer_size, self.hidden_layer_size, self.dropout_rate
            )

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ("W2" in name or "W3" in name) and "bias" not in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif ("skip_linear" in name or "W1" in name) and "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            elif "bias" in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        if self.additional_context:
            x, context = x
            n2 = self.ELU(self.W2(x) + self.W3(context))
        else:
            n2 = self.ELU(self.W2(x))

        n1 = self.W1(n2)

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        return output


class ScaledDotProductAttention(pl.LightningModule):
    def __init__(self, dropout=0.0, scale=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # print('---Inputs----')
        # print('q: {}'.format(q[0]))
        # print('k: {}'.format(k[0]))
        # print('v: {}'.format(v[0]))

        attn = torch.bmm(q, k.permute(0, 2, 1))
        # print('first bmm')
        # print(attn.shape)
        # print('attn: {}'.format(attn[0]))

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension
        #    print('attn_scaled: {}'.format(attn[0]))

        if mask is not None:
            # fill = torch.tensor(-1e9).to(DEVICE)
            # zero = torch.tensor(0).to(DEVICE)
            attn = attn.masked_fill(mask == 0, -1e9)
        #    print('attn_masked: {}'.format(attn[0]))

        attn = self.softmax(attn)
        # print('attn_softmax: {}'.format(attn[0]))
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class InterpretableMultiHeadAttention(pl.LightningModule):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)]
        )
        self.v_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_v, bias=False) for _ in range(self.n_head)]
        )
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            # print('qs layer: {}'.format(qs.shape))
            head, attn = self.attention(qs, ks, vs, mask)
            # print('head layer: {}'.format(head.shape))
            # print('attn layer: {}'.format(attn.shape))
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        # print('concat heads: {}'.format(head.shape))
        # print('heads {}: {}'.format(0, head[0,0,Ellipsis]))
        attn = torch.stack(attns, dim=2)
        # print('concat attn: {}'.format(attn.shape))

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # print('outputs mean: {}'.format(outputs.shape))
        # print('outputs mean {}: {}'.format(0, outputs[0,0,Ellipsis]))
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class VariableSelectionNetwork(pl.LightningModule):
    def __init__(
        self,
        hidden_layer_size,
        dropout_rate,
        output_size,
        input_size=None,
        additional_context=None,
    ):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        self.flattened_grn = GatedResidualNetwork(
            self.hidden_layer_size,
            input_size=self.input_size,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
            additional_context=self.additional_context,
        )

        self.per_feature_grn = nn.ModuleList(
            [
                GatedResidualNetwork(
                    self.hidden_layer_size, dropout_rate=self.dropout_rate
                )
                for _ in range(self.output_size)
            ]
        )

    def forward(self, x):
        #####################
        # Non Static Inputs #
        #####################
        if self.additional_context:
            embedding, static_context = x
            # print("static_context")
            # print(static_context.shape)

            time_steps = embedding.shape[1]
            batch_size = embedding.shape[0]
            # print("embedding")
            # print(embedding.shape)
            flatten = embedding.view(
                -1,
                time_steps,
                self.hidden_layer_size * self.output_size,  # flattens the input
            )
            # print("flatten")
            # print(flatten.shape)

            static_context = static_context.unsqueeze(1)
            # print("static_context")
            # print(static_context.shape)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = self.flattened_grn((flatten, static_context))
            # print("mlp_outputs")
            # print(mlp_outputs.shape)

            sparse_weights = F.softmax(mlp_outputs, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(2)
            # print("sparse_weights")
            # print(sparse_weights.shape)

            trans_emb_list = []
            for i in range(self.output_size):
                e = self.per_feature_grn[i](
                    embedding.view(batch_size, time_steps, self.hidden_layer_size, -1)[
                        Ellipsis, i
                    ]
                )
                trans_emb_list.append(e)
            transformed_embedding = torch.stack(trans_emb_list, axis=-1)
            # print("transformed_embedding")
            # print(transformed_embedding.shape)

            combined = sparse_weights * transformed_embedding
            # print("combined")
            # print(combined.shape)

            temporal_ctx = torch.sum(combined, dim=-1)
            # print("temporal_ctx")
            # print(temporal_ctx.shape)

        # Static Inputs
        else:
            embedding = x
            # print("embedding")
            # print(embedding.shape)

            flatten = torch.flatten(embedding, start_dim=1)
            # flatten = embedding.view(batch_size, -1)
            # print("flatten")
            # print(flatten.shape)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = self.flattened_grn(flatten)
            # print("mlp_outputs")
            # print(mlp_outputs.shape)

            sparse_weights = F.softmax(mlp_outputs, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(-1)
            # print("sparse_weights")
            # print(sparse_weights.shape)

            trans_emb_list = []
            for i in range(self.output_size):
                # print('embedding for the per feature static grn')
                # print(embedding[Ellipsis, i].shape)
                e = self.per_feature_grn[i](
                    embedding.view(-1, self.output_size, self.hidden_layer_size)[
                        :, i : i + 1, :
                    ]
                )
                trans_emb_list.append(e)
            transformed_embedding = torch.cat(trans_emb_list, axis=1)
            # print("transformed_embedding")
            # print(transformed_embedding.shape)

            combined = sparse_weights * transformed_embedding
            # print("combined")
            # print(combined.shape)

            temporal_ctx = torch.sum(combined, dim=1)
            # print("temporal_ctx")
            # print(temporal_ctx.shape)

        return temporal_ctx, sparse_weights
