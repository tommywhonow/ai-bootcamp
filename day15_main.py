import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# 1. DATASET CLASS
class HiringDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
# 2. CREATE DATA
np.random.seed(42)
n           = 500
age         = np.random.randint(22, 60, n).astype(float)
income      = np.random.randint(20000, 100000, n).astype(float)
education   = np.random.randint(8, 22, n).astype(float)
experience  = np.random.randint(0, 20, n).astype(float)
hired = (0.3 * education + 0.5 * experience 
         + np.random.normal(0, 2, n) > 10).astype(float)\

X_np = np.column_stack([age, income, education, experience])
y_np = hired.reshape(-1, 1)
X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0) # Normalise

X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
X_train = torch.tensor(X_tr, dtype=torch.float32)
X_test  = torch.tensor(X_te, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.float32)
y_test  = torch.tensor(y_te, dtype=torch.float32)

# 3. DATALOADERS
train_loader = DataLoader(HiringDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(HiringDataset(X_test, y_test),   batch_size=32, shuffle=False)

# 4. NETWORK WITH DROPOUT
class HiringNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1  = nn.Linear(4, 16)
        self.layer2  = nn.Linear(16, 8)
        self.layer3  = nn.Linear(8, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=.03)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x)) # 4 -> 16 relu
        x = self.dropout(x)
        x = self.relu(self.layer2(x)) # 16 -> 8 relu
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x)) # 8  -> 1 relu
        return x
    
# 5. SETUP
torch.manual_seed(42)
model = HiringNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # gradient descent
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# 6. TRAINING LOOP
print("Training with mini batches...")
start=time.time()
for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    for bX, by in train_loader:
        optimizer.zero_grad()               # Zero Gradients
        loss = criterion(model(bX), by)     # Compute Loss
        loss.backward()                     # Backward pass - autograd computes all gradients
        optimizer.step()                    # Update Weights
        epoch_loss += loss.item()
    if (epoch + 1) % 10 ==0:
        model.eval()                # Switches to evaluation mode — disables dropout before measuring accuracy.
        correct = sum(              # Counts how many test samples the model predicted correctly. 
        
            (model(bX) > 0.5).float().eq(by).sum().item()
            for bX, by in test_loader
        )                           # sum() adding up correct predictions across all test batches.
        
        acc = correct / len(y_test)  # Divides correct predictions by total test samples → accuracy as a fraction.
        # Every 10 epochs:
            # switch to eval mode
            # loop through all test batches
            # count correct predictions
            # divide by total samples
            #print accuracy
        print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.3f}")
        model.train()  
end = time.time()
print(f"\nTraining took {end - start:.2f} seconds")

# 7. FINAL EVALUATION
model.eval()
with torch.no_grad():
    correct = sum((model(bX)>0.5).float().eq(by).sum().item() for bX, by in test_loader)
final_acc = correct / len(y_test)
print(f"\nFinal accuracy: {final_acc:.3f}   ({correct} /{len(y_test)} )")

# 8. SAVE
torch.save(model.state_dict(), 'hiring_model.pth')
print("Model saved to hiring_model.pth")

# 9. LOAD AND VERIFY
loaded = HiringNet()
loaded.load_state_dict(torch.load('hiring_model.pth', map_location='cpu'))
loaded.eval()
with torch.no_grad():
    correct2 = sum((loaded(bX)>0.5).float().eq(by).sum().item() for bX, by in test_loader)
loaded_acc = correct2 / len(y_test)
print(f"Loaded model accuracy: {loaded_acc:.3f}")
print(f"Matches original:       {abs(final_acc - loaded_acc) < 0.001}")