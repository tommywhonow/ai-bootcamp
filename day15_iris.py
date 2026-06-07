import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATA
class IrisDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y
    def __len__(self) -> int:
        return int(self.X.shape[0])
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
# LOAD REAL DATA
iris = load_iris() 
X_np = iris.data.astype(np.float32)  # type: ignore[attr-defined]
y_np = iris.target  # type: ignore[attr-defined] # int 0, 1, 2   

X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
scaler = StandardScaler()   
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# Float 32 for features, long (integer) for labels - required by CrossEntropy
X_train = torch.tensor(X_tr, dtype=torch.float32)
X_test  = torch.tensor(X_te, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.long)
y_test  = torch.tensor(y_te, dtype=torch.long)

# DATALOADERS
train_loader = DataLoader(IrisDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader  = DataLoader(IrisDataset(X_test, y_test),   batch_size=16, shuffle=False)
print(f"Training samples: {len(X_train)}, Test samples {len(X_test)}")
print(f"Training batches: {len(train_loader)}")

# NETWORK -3 output neurons, NO sigmoid on output
class IrisNet(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 3) # 3 outputs - one per class
        self.relu   = nn.ReLU()
        # NO SIGMOID - CrossEntropyLoss handles it internally
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)      # raw logits - no activation
        return x

# SET UP
torch.manual_seed(42)
model = IrisNet()
criterion = nn.CrossEntropyLoss()   # handles sigmoid internally
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# TRAINING LOOP
print("\nTraining on Iris...")
model.train()
for epoch in range(100):
    epoch_loss = 0.0
    for bX, by in train_loader:
        optimizer.zero_grad()
        output = model(bX)          # shape(batch, 3)
        loss   = criterion(output, by) # by must be long
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg = epoch_loss / len(train_loader)
        print(f"    Epoch {epoch+1:3d} | Avg Loss: {avg:4f}")

# EVALUATE
model.eval()
correct = 0
with torch.no_grad():
    for bX, by in test_loader:
        predicted = torch.argmax(model(bX), dim=1)
        correct  += (predicted == by).sum().item()
accuracy = correct / len(y_test)
print(f"\nTest accuracy: {accuracy:.3f}    ({correct}/{len(y_test)})")

# SAVE
torch.save(model.state_dict(), 'iris_model.pth')
print("Model saved to iris_model.pth")

# LOAD AND VERIFY
loaded = IrisNet()
loaded.load_state_dict(torch.load('iris_model.pth', map_location='cpu'))
loaded.eval()
c2 = 0
with torch.no_grad():
    for bX,by in test_loader:
        c2 = int((torch.argmax(loaded(bX), dim=1)==by).sum().item() )
    print(f"Loaded accuracy: {c2/len(y_test):.3f}")
    print(f"Matches          {abs(accuracy - c2/len(y_test) < 0.001)}")