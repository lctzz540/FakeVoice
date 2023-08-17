import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.fc = nn.Linear(32 * (output_dim // 4), output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=16, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2
        )
        self.fc = nn.Linear(32 * (input_dim // 8), 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x
