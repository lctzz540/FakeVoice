import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels, hidden_dim=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                latent_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                hidden_dim * 2, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)
