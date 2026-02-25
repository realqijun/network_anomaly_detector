import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, width = x.size()
        q = self.query(x).view(batch, -1, width).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, width)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(batch, -1, width)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        return self.gamma * out.view(batch, channels, width) + x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return F.gelu(x + self.conv(x))


class ConvAutoencoder(nn.Module):
    def __init__(self, n_features, window_size=10, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.flat_dim = 32 * (window_size // 2)

        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            ResidualBlock(64),
            AttentionBlock(64),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.GELU(),
            nn.Unflatten(1, (32, window_size // 2)),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock(64),
            nn.ConvTranspose1d(64, n_features, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
