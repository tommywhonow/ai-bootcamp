import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms    # type: ignore[import-untyped]

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PREPARE DATA - ToTensor converts images to tensors, pixels scaled 0-1
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
print(f"Training images: {len(train_data)}")    #60000
print(f"Test images:     {len(test_data)}")     #10000

# DEFINE NETWORK
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)     # 1 -> 16 feature maps
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding =1)    # 2 -> 32 featyre maps
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        # After 2 pools: 28 -> 14 -> 7 so 32 channels x 7 x 7 = 1568
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)     # 10 digit class
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))     # 28 x 28 -> 14 x 14
        x = self.pool(self.relu(self.conv2(x)))     # 14 -> 7
        x = x.view(x.size(0), -1)                   # flatten -> (batch, 1568)
        x = self.relu(self.fc1(x))                  # linear + relu
        x = self.fc2(x)                             #output logits (no actuvation)
        return x

# SETUP
torch.manual_seed(42)
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# TRAIN
print("Training...")
for epoch in range(3):
    model.train()
    epoch_loss = 0.0
    for bX, by in train_loader:
        bX, by = bX.to(device), by.to(device)
        optimizer.zero_grad()
        output = model(bX)
        loss   = criterion(output, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"    Epoch {epoch+1} | Avg Loss: {epoch_loss/len(train_loader):.4f}")

# EVALUATE
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for bX, by in test_loader:
        bX, by = bX.to(device), by.to(device)
        predicted = torch.argmax(model(bX), dim=1)
        correct  += int((predicted == by).sum().item())
        total    += by.size(0)
print(f"\nTest accuracy {correct/total:.3f}    ({correct}/{total})")

# SAVE 
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved as mnist_cnn.pth")