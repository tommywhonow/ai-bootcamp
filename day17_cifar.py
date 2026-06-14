import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# DEVICES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PREPARE DATA - augmentation ontrain, plian on test
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data   = datasets.CIFAR10(root='./data',  train=True,  download=True, transform=train_transform)
test_data    = datasets.CIFAR10(root='./data',  train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
print(f"Training images: {len(train_data)}")    # 50000
print(f"Test images      {len(test_data)}")     # 10000

# DEFINE NETWORK - 3 conv blocks
class CNN(nn.Module):
    """
    CNN for CIFAR-10 image classification.

    Architecture:
    Input:  3×32×32 (RGB image)
    Conv1:  3→32 channels, 32×32 (same padding)
    Pool1:  32×32 → 16×16
    Conv2:  32→64 channels, 16×16
    Pool2:  16×16 → 8×8
    Conv3:  64→128 channels, 8×8
    Pool3:  8×8 → 4×4
    Flatten: 128×4×4 = 2048
    FC1:    2048 → 256
    FC2:    256 → 10 (class scores)
    """

    def __init__(self) -> None:
        super().__init__()
        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        # 32 -> 16 -> 8 -> 4 after 3 pools, 128 channels: 128*4*4 = 2048
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # 32 -> 16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # 16 -> 8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)                           # flatten -> 2048
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x= self.fc2(x)
        return x
    
# SETUP
torch.manual_seed(42)
model     = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# TRAIN - use 5 epochs on CPU, 15+ on GPU
epochs = 15
print("Training...")
for epoch in range(epochs):
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
    print(f"    Epoch {epoch+1:2d} | Avg Loss: {epoch_loss/len(train_loader):.4f}")

# EVALUATE
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for bx, by in test_loader:
        bx, by = bX.to(device), by.to(device)
        predicted = torch.argmax(model(bX), dim=1)
        correct  += int((predicted == by).sum().item())
        total    += by.size(0)
print(f"\nTest accuracy: {correct/total:.3f}   ({correct}/{total})")

# SAVE
torch.save(model.state_dict(), 'cifar_cnn.pth')
print("Model saved to cifar_cnn.pth")