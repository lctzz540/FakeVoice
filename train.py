import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import VoiceDataset
from nets import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 8
output_duration_ms = 20000
batch_size = 32
num_epochs = 100
learning_rate = 0.0001


def transform(audio):
    return torch.tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)


data_directory = "./data"
voice_dataset = VoiceDataset(
    data_directory, output_duration_ms, transform=transform)

dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=True)

generator = Generator(input_dim, output_duration_ms).to(device)
discriminator = Discriminator(output_duration_ms).to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, real_data in enumerate(dataloader):
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        optimizer_D.zero_grad()
        real_output = discriminator(real_data.to(device))
        d_real_loss = criterion(real_output, real_labels)

        noise = torch.randn(batch_size, input_dim,
                            output_duration_ms, device=device)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        d_fake_loss = criterion(fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}"
            )

    with torch.no_grad():
        noise = torch.randn(32, input_dim, output_duration_ms, device=device)
        fake_samples = generator(noise)
