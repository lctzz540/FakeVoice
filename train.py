import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import VoiceConversionModel
from data import VoiceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_dim = 514
hidden_dim = 64
output_dim = 514
learning_rate = 0.001
batch_size = 16

conversion_model = VoiceConversionModel(
    input_dim, hidden_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(conversion_model.parameters(), lr=learning_rate)


def transform(audio):
    return torch.tensor(audio, dtype=torch.float32).to(device)


dataset = VoiceDataset(data_folder="./data", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        predicted_data = conversion_model(batch_data)

        loss = criterion(predicted_data, batch_data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
torch.save(
    conversion_model.state_dict(),
    "./weight/conversion_model_weights.pth",
)
