import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim * 2),
            ConvBlock(hid_dim * 2, hid_dim * 2),
            ConvBlock(hid_dim * 2, hid_dim * 4),
        )

        self.attention = nn.MultiheadAttention(embed_dim=hid_dim * 4, num_heads=8, batch_first=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 4, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = X.transpose(1, 2)  # [b, t, d] に変換
        X, _ = self.attention(X, X, X)
        X = X.transpose(1, 2)  # [b, d, t] に戻す

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)


import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Linear(in_channels, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, num_classes)
        self.fc3 = nn.Linear(hid_dim, hid_dim)

        self._init_weights()

    def _init_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)
              if m.bias is not None:
                  nn.init.zeros_(m.bias)
          elif isinstance(m, nn.LayerNorm):
              nn.init.ones_(m.weight)
              nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)  # (batch_size, in_channels, seq_len)
        X = self.embedding(X)  # (batch_size, seq_len, hid_dim)
        X = self.layernorm1(X)  # (batch_size, seq_len, hid_dim)
        X = self.pos_encoder(X)  # (batch_size, seq_len, hid_dim)
        X = self.transformer(X)  # (batch_size, seq_len, hid_dim)
        X = X.mean(dim=1)  # グローバル平均プーリング (batch_size, hid_dim)
        X = self.layernorm1(X) # added
        X = self.dropout(X)
        # X = self.fc3(X) # added
        # X = self.dropout(X) # added
        # X = self.layernorm1(X) # added
        X = self.fc2(X)  # (batch_size, num_classes)

        return X
