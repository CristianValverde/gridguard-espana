"""
Module-level LSTM Autoencoder nn.Module.

Lives here (not inside a function) so pickle can serialize it via
``ml.train.lstm_model.LSTMAutoencoderModule`` — a requirement for
``torch.save`` / ``torch.load`` to work correctly.
"""
import torch.nn as nn


class LSTMAutoencoderModule(nn.Module):
    """Encoder-decoder LSTM for unsupervised anomaly detection."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        seq_len = x.size(1)
        _, (h_n, c_n) = self.encoder(x)
        # Repetir el estado oculto final para cada paso del decoder
        context = h_n[-1].unsqueeze(1).expand(-1, seq_len, -1)
        decoded, _ = self.decoder(context, (h_n, c_n))
        return self.output_layer(decoded)
