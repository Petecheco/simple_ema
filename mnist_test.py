import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ema import EMAModel
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Init the ema model
ema = EMAModel(model, beta=0.999, device=device)
accuracy_emas = []
accuracy_raws = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (data, target) in train_loop:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        if batch_idx % 100 == 0:
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()
    ema_model = ema.ema_model
    correct_ema = 0
    correct_raw = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs_ema = ema_model(data)
            outputs_raw = model(data)
            _, predicted_ema = torch.max(outputs_ema.data, 1)
            _, predicted_raw = torch.max(outputs_raw.data, 1)
            total += target.size(0)
            correct_ema += (predicted_ema == target).sum().item()
            correct_raw += (predicted_raw == target).sum().item()
    accuracy_ema = correct_ema / total
    accuracy_emas.append(accuracy_ema)
    accuracy_raw = correct_raw / total
    accuracy_raws.append(accuracy_raw)
    print(f"Epoch [{epoch + 1}/{num_epochs}], EMA Model Accuracy: {100 * accuracy_ema:.2f}%, Raw Model Accuracy: {100 * accuracy_raw:.2f}%")

with open("./accuracy_emas.pkl", "wb") as f:
    pickle.dump(accuracy_emas, f)

with open("./accuracy_raws.pkl", "wb") as f:
    pickle.dump(accuracy_raws, f)