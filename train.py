import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import VoiceDataset
from nets import Discriminator, Generator

batch_size = 4
num_epochs = 100
learning_rate = 0.0002
latent_dim = 100
fixed_duration_ms = 3000
output_channels = 1
hidden_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(latent_dim, output_channels, hidden_dim).to(device)
discriminator = Discriminator(output_channels, hidden_dim).to(device)

optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()


def transform(audio):
    return torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)


dataset = VoiceDataset(
    data_folder="./data", fixed_duration_ms=fixed_duration_ms, transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

for epoch in range(num_epochs):
    for real_audio_batch in dataloader:
        real_audio_batch = real_audio_batch.unsqueeze(1).to(device)

        optimizer_discriminator.zero_grad()
        real_labels = torch.ones(real_audio_batch.size(0), 16534).to(device)
        fake_labels = torch.zeros(
            real_audio_batch.size(0), real_audio_batch.size(2)
        ).to(device)

        real_outputs = discriminator(real_audio_batch)
        noise = torch.randn(
            real_audio_batch.size(0), latent_dim, real_audio_batch.size(2)
        ).to(device)
        fake_audio_batch = generator(noise)
        fake_outputs = discriminator(fake_audio_batch.detach())

        real_outputs = real_outputs.view(real_outputs.size(0), -1)
        fake_outputs = fake_outputs.view(fake_outputs.size(0), -1)

        loss_real = criterion(real_outputs, real_labels)
        loss_fake = criterion(fake_outputs, fake_labels)
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        optimizer_generator.zero_grad()
        noise = torch.randn(real_audio_batch.size(0), latent_dim, 16534).to(device)
        fake_audio_batch = generator(noise)
        fake_outputs = discriminator(fake_audio_batch)
        loss_generator = criterion(fake_outputs.squeeze(), real_labels)
        loss_generator.backward()
        optimizer_generator.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] Discriminator Loss: {loss_discriminator.item():.4f} Generator Loss: {loss_generator.item():.4f}"
    )
