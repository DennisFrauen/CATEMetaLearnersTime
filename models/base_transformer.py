import math
import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from abc import ABC, abstractmethod


class EncoderTransformer(L.LightningModule, ABC):
    def __init__(self, config):
        """
        :param input_dim: int: Dimension of input features (i.e., vitals+prev_outputs+prev_treatments combined)
        :param time_dim: int: Length of time sequence
        :param d_model: int: Dimension of embeddings
        :param num_heads: int: Number of attention heads
        :param num_encoder_layers: int: Number of encoder layers
        :param dim_feedforward: int: Dimension of feedforward network within EncoderLayer
        :param dropout: float: Dropout rate
        :param dim_output_hidden_units: int: Dimension of output hidden units
        :param output_type: str: Type of output. Values: 'regression_type1' / 'regression_type2' / 'classification'
        :param lr: float: Learning rate
        :param output_type: string: Regression/ Classification
        """
        input_dim = config["input_dim"]
        self.d_model = config["d_model"]
        time_dim = config["time_dim"]
        num_heads = config["num_heads"]
        num_encoder_layers = config["num_encoder_layers"]
        dim_feedforward = config["dim_feedforward"]
        self.dropout = config["dropout"]
        self.dim_output_hidden_units = config["dim_output_hidden_units"]
        self.output_type = config["output_type"]
        self.lr = config["lr"]
        self.tau = config["tau"]
        self.l2_reg = config["l2_reg"]

        super(EncoderTransformer, self).__init__()

        self.input_layer = nn.Linear(input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(time_dim, self.d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, num_heads, dim_feedforward, self.dropout, batch_first=True),
            num_encoder_layers, norm=nn.LayerNorm(self.d_model))
        self.causal_mask = self.generate_causal_mask(time_dim - self.tau)

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        def weighted_mse(input, target, weight):
            return torch.mean(weight * (input - target) ** 2)

        def mse(input, target, weight):
            return torch.mean((input - target) ** 2)

        def cross_entropy(input, target, weight):
            return F.binary_cross_entropy_with_logits(input, target, reduction='mean')

        if self.output_type == 'regression':
            self.loss = mse
        elif self.output_type == 'weighted_regression':
            self.loss = weighted_mse
        elif self.output_type == 'classification':
            self.loss = cross_entropy
        else:
            raise ValueError(f'output_type={self.output_type} not supported')

    @abstractmethod
    def forward(self, batch):
        pass

    def history_representation(self, covariates, prev_outcomes, prev_treatments, active_entries):
        z = self.input_layer(torch.concat((covariates, prev_outcomes, prev_treatments), dim=-1))
        z = self.positional_encoding(z)
        z = self.transformer_encoder(z,
                                     is_causal=True,  # sequential data
                                     mask=self.causal_mask,  # self-attention mask
                                     src_key_padding_mask=active_entries.squeeze(dim=-1) == 0  # padding mask
                                     )
        return z

    def training_step(self, batch):
        self.train()
        preds, targets, weights = self(batch)
        loss = self.loss(preds, targets, weights)
        self.log_dict({"train_loss": loss}, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch):
        self.eval()
        preds, targets, weights = self(batch)
        loss = self.loss(preds, targets, weights)
        self.log_dict({"val_loss": loss}, logger=True, on_epoch=True, on_step=False)
        return loss

    @abstractmethod
    def predict_step(self, batch):
        pass

    def configure_optimizers(self):
        params = self.parameters()
        return optim.Adam(params, lr=self.lr, weight_decay=self.l2_reg)

    @staticmethod
    def generate_causal_mask(seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        return mask.to(torch.bool)  # True indicates masked

    def fit(self, data_loader, name, logging=False):
        data_loader.batch_size = {"train": self.batch_size, "val": self.batch_size, "test": self.batch_size}
        if logging:
            logger = WandbLogger(project='tsmetalearners', name=name)
        else:
            logger = False
        trainer = L.Trainer(max_epochs=self.epochs, enable_progress_bar=False, enable_model_summary=False,
                            accelerator="auto", logger=logger, enable_checkpointing=False)
        trainer.fit(self, data_loader)
        val_results = trainer.validate(self, data_loader)

        if logging:
            logger.experiment.finish()
        return val_results


class PositionalEncoding(L.LightningModule):
    # Positional encoding along time-dimension
    def __init__(self, time_dim, d_model):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(time_dim, d_model)
        position = torch.arange(0, time_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Remove the transpose
        # Non-trainable positional encoding
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Adjust the indexing
        return self.dropout(x)


class Transformer_History(EncoderTransformer):
    # Transformer model conditioning on history only
    def __init__(self, config):
        super().__init__(config)

        self.output_head = nn.Sequential(nn.Linear(self.d_model, self.dim_output_hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(self.dim_output_hidden_units, 1))

    def forward(self, batch):

        targets, covariates, prev_outcomes, prev_treatments, future_treatments, active_entries, weights = batch

        z = self.history_representation(covariates, prev_outcomes, prev_treatments, active_entries)

        if self.output_type == 'classification':
            targets = future_treatments[:, :, 0:1]

        preds = self.output_head(z)
        return preds * active_entries, targets * active_entries, weights * active_entries

    def predict_step(self, batch):
        self.eval()
        preds, _, _ = self(batch)

        if self.output_type == 'classification':
            preds = torch.sigmoid(preds)

        return preds


class Transformer_History_Treatment(EncoderTransformer):
    # Transformer model conditioning on history and current + future treatments
    # Allows for treatment intervention during prediction step
    def __init__(self, config):
        super().__init__(config)
        self.output_head = nn.Sequential(nn.Linear(self.d_model + self.tau + 1, self.dim_output_hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(self.dim_output_hidden_units, 1))

    def forward(self, batch):
        targets, covariates, prev_outcomes, prev_treatments, future_treatments, active_entries, weights = batch

        z = self.history_representation(covariates, prev_outcomes, prev_treatments, active_entries)
        z = torch.concat((z, future_treatments[:, :, :self.tau+1]), dim=-1)

        preds = self.output_head(z)
        return preds * active_entries, targets * active_entries, weights * active_entries

    def predict_step(self, batch, a_int=None):
        self.eval()
        # a_int: list of interventions
        targets, covariates, prev_outcomes, prev_treatments, future_treatments, active_entries, weights = batch

        if a_int is not None:
            future_interventions = torch.tensor(a_int, dtype=torch.float32).repeat(targets.size(0), targets.size(1), 1)
        else:
            future_interventions = future_treatments[:, :, :self.tau+1]

        z = self.history_representation(covariates, prev_outcomes, prev_treatments, active_entries)
        z = torch.concat((z, future_interventions), dim=-1)

        preds = self.output_head(z)

        if self.output_type == 'classification':
            preds = torch.sigmoid(preds)

        return preds
