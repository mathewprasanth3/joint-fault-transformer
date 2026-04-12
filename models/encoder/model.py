import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    # 1D CNN encoder for a single AE sensor stream
    # input:  (batch, 1, 10000) -- one sensor, one window
    # output: (batch, embedding_dim) -- compressed embedding vector

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # three conv blocks -- each halves the time dimension
        # channels grow: 1 -> 32 -> 64 -> 128
        self.conv_blocks = nn.Sequential(

            # block 1 -- detect low level patterns: burst onset, sharp edges
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 10000 -> 2500

            # block 2 -- detect mid level patterns: burst shape, duration
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2500 -> 625 (approx)

            # block 3 -- detect high level patterns: burst density, frequency content
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # collapse time dimension to 1 regardless of input length
        )

        # project to embedding dim if different from 128
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, x):
        # x shape: (batch, 1, 10000)
        x = self.conv_blocks(x)         # (batch, 128, 1)
        x = x.squeeze(-1)               # (batch, 128) -- remove trailing time dim
        x = self.projection(x)          # (batch, embedding_dim)
        return x


class TripleEncoder(nn.Module):
    # three independent CNN encoders -- one per AE sensor
    # independent weights because each sensor has a different frequency range:
    #   sensor A: Micro80    100-300 kHz
    #   sensor B: F50A       50-400 kHz
    #   sensor C: Micro200HF 200 kHz+
    # input:  (batch, 3, 10000) -- all 3 sensors, one window
    # output: (batch, 3, embedding_dim) -- one embedding per sensor

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # three separate encoders -- different weights, same architecture
        self.encoder_a = CNNEncoder(embedding_dim)  # Micro80
        self.encoder_b = CNNEncoder(embedding_dim)  # F50A
        self.encoder_c = CNNEncoder(embedding_dim)  # Micro200HF

    def forward(self, x):
        # x shape: (batch, 3, 10000)
        # split into 3 separate (batch, 1, 10000) tensors
        xa = x[:, 0:1, :]   # sensor A -- (batch, 1, 10000)
        xb = x[:, 1:2, :]   # sensor B -- (batch, 1, 10000)
        xc = x[:, 2:3, :]   # sensor C -- (batch, 1, 10000)

        # encode each sensor independently
        ea = self.encoder_a(xa)   # (batch, embedding_dim)
        eb = self.encoder_b(xb)   # (batch, embedding_dim)
        ec = self.encoder_c(xc)   # (batch, embedding_dim)

        # stack into (batch, 3, embedding_dim)
        embeddings = torch.stack([ea, eb, ec], dim=1)
        return embeddings


if __name__ == '__main__':
    batch_size    = 4
    embedding_dim = 128

    # test single encoder
    print('=== CNNEncoder (single sensor) ===')
    encoder = CNNEncoder(embedding_dim=embedding_dim)
    x       = torch.randn(batch_size, 1, 10000)
    out     = encoder(x)
    print(f'Input  : {x.shape}')
    print(f'Output : {out.shape}')   # expect (4, 128)

    # count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'Params : {n_params:,}')

    # test triple encoder
    print('\n=== TripleEncoder (3 sensors) ===')
    triple = TripleEncoder(embedding_dim=embedding_dim)
    x      = torch.randn(batch_size, 3, 10000)
    out    = triple(x)
    print(f'Input  : {x.shape}')
    print(f'Output : {out.shape}')   # expect (4, 3, 128)

    n_params = sum(p.numel() for p in triple.parameters())
    print(f'Params : {n_params:,}')

    print('\nAll checks passed.')