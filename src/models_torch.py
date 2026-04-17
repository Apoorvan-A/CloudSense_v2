"""
models_torch.py
===============
PyTorch implementations of all models used in the comparative study.

Models
------
1.  LSTMModel          – Standard LSTM  (Bansal & Kumar, 2023)
2.  CNNLSTMModel       – 1-D Conv + LSTM (Bi et al., 2023)
3.  BiLSTMModel        – Bidirectional LSTM (Xing, 2024; Vasumathi, 2025)
4.  TransformerModel   – Transformer encoder (Lackinger et al., 2024)
5.  CEEMDANBiLSTM      – PROPOSED: CEEMDAN decomposition + CNN-BiLSTM ensemble

All models accept input of shape  (batch, look_back, 1)  and output (batch, 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#  1.  Standard LSTM  —  Bansal & Kumar (2023)
class LSTMModel(nn.Module):
    """
    Stacked LSTM network for cloud workload prediction.
    Reference: Bansal & Kumar, IEEE Trans. Cloud Computing, 2023.
    """
    name = "LSTM (Bansal 2023)"

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, look_back=48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)          # (B, T, H)
        out     = self.dropout(out[:, -1, :])  # last time step
        return self.fc(out)            # (B, 1)


#  2.  CNN-LSTM  —  Bi et al. (2023)
class CNNLSTMModel(nn.Module):
    """
    1-D Convolutional feature extractor followed by a stacked LSTM.
    The CNN captures local temporal patterns; the LSTM models long-range deps.
    Reference: Bi, Ma, Yuan, Zhang — IEEE Trans. Sustainable Computing, 2023.
    """
    name = "CNN-LSTM (Bi 2023)"

    def __init__(self, input_size=1, cnn_filters=32, kernel_size=3,
                 hidden_size=64, num_layers=2, dropout=0.2, look_back=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size  = cnn_filters,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, 1) → permute for Conv1d → (B, 1, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)               # (B, filters, T)
        x = x.permute(0, 2, 1)         # (B, T, filters)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


#  3.  Bidirectional LSTM  —  Xing (2024) / Vasumathi (2025)
class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM: processes the sequence in both directions, then
    concatenates hidden states for richer temporal context.
    References: Xing, Journal of Grid Computing, 2024;
                Vasumathi et al., Expert Systems Conf., 2025.
    """
    name = "Bi-LSTM (Xing 2024)"

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, look_back=48):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),   # *2 for bidirectional concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.bilstm(x)        # (B, T, H*2)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


#  4.  Transformer Encoder  —  Lackinger et al. (2024)
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """
    Transformer encoder for time-series forecasting.
    Uses multi-head self-attention to capture long-range temporal dependencies.
    Reference: Lackinger, Morichetta, Dustdar — IEEE SCC, 2024.
    """
    name = "Transformer (Lackinger 2024)"

    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, look_back=48):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=look_back + 10,
                                             dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)          # (B, T, d_model)
        return self.fc(x[:, -1, :])  # use last token


#  5.  CEEMDAN + CNN-BiLSTM  —  PROPOSED MODEL
class IMFSubModel(nn.Module):
    """
    Small CNN-BiLSTM sub-model trained on a single CEEMDAN IMF component.
    Lightweight to keep per-IMF training fast while still capturing patterns.
    """
    def __init__(self, look_back=48, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size   = 16,
            hidden_size  = hidden,
            num_layers   = 1,
            batch_first  = True,
            bidirectional= True,
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (B, T, 1)
        c = x.permute(0, 2, 1)             # (B, 1, T)
        c = self.conv(c).permute(0, 2, 1)  # (B, T, 16)
        out, _ = self.bilstm(c)
        return self.fc(out[:, -1, :])       # (B, 1)


class CEEMDANBiLSTM(nn.Module):
    """
    Proposed Architecture: CEEMDAN + CNN-BiLSTM Ensemble.

    Pipeline
    --------
    1. CEEMDAN decomposes the raw signal into K IMFs + residue.
    2. One IMFSubModel is trained independently per IMF.
    3. Final prediction = sum of all per-IMF predictions.

    This approach explicitly removes noise before modelling, enabling each
    sub-model to learn a simpler, more predictable frequency component.

    Reference: Torres et al. (2011) CEEMDAN + proposed CNN-BiLSTM integration.
    """
    name = "CEEMDAN+CNN-BiLSTM (Proposed)"

    def __init__(self, n_imfs=6, look_back=48, hidden=32):
        super().__init__()
        self.n_imfs    = n_imfs
        self.look_back = look_back
        # One sub-model per IMF component
        self.sub_models = nn.ModuleList(
            [IMFSubModel(look_back=look_back, hidden=hidden)
             for _ in range(n_imfs)]
        )

    def forward(self, x_list):
        """
        x_list : list of n_imfs tensors, each (B, T, 1).
        Returns : (B, 1) – summed prediction.
        """
        total = None
        for i, (sub, xi) in enumerate(zip(self.sub_models, x_list)):
            pred = sub(xi)
            total = pred if total is None else total + pred
        return total
