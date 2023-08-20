from torch import nn


class VoiceConversionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VoiceConversionModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim,
                               num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim,
                               num_layers=2, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded
