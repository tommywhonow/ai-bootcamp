import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# CREATE DATA - same as day 13
np.random.seed(42)
n =500
age         = np.random.randint(18, 60, n).astype(float)
income      = np.random.randint(20000, 100000, n).astype(float)
education   = np.random.randint(8, 22, n).astype(float)
experience  = np.random.randint(0, 20, n).astype(float)
hired       = (0.3 * education + 0.5 * experience
               +np.random.normal(0, 2, n)>10).astype(float)

X_np = np.column_stack([age, income, education, experience])
y_np = hired.reshape(-1, 1)

# Normalise
X_np = (X_np -X_np.mean(axis=0)) / X_np.std(axis=0)

# Split
X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Covert to Pytorch tensors - neural networks
X_train = torch.tensor(X_tr, dtype=torch.float32)
X_test  = torch.tensor(X_te, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.float32)
y_test  = torch.tensor(y_te, dtype=torch.float32)

# DEFINE NETWORK
class HiringNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Same arithmatic as Day 13: 4 -> 8 -> 4 -> 1
        self.layer1  = nn.Linear(4, 8)
        self.layer2  = nn.Linear(8, 4)
        self.layer3  = nn.Linear(4, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))    # 4 -> 8 relu
        x = self.relu(self.layer2(x))    # 8 -> 4 relu
        x = self.sigmoid(self.layer3(x)) # 4 -> 1 sigmoid
        return x
    
# SETUP
torch.manual_seed(42)
model     = HiringNet()
criterion = nn.BCELoss() # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.01) # gradient descent

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print(f"Architecture: {model}\n")

# TRAINING LOOP
print("Training...")
for epoch in range(1000):
    # 1. Zero gradients - alwyas first
    optimizer.zero_grad()

    # 2. Forward pass
    output = model(X_train)

    # 3. Compute loss
    loss = criterion(output, y_train)
    
    # 4. Backward pass - autograd computes all gradients
    loss.backward()

    # 5. Update weights
    optimizer.step()

    if (epoch + 1) %100 == 0:
        print(f"    Epoch {epoch+1:4d} | Loss: {loss.item():.4f}")

# EVALUATE
model.eval()    # set model to evaluate mode
with torch.no_grad():
    predictions = model(X_test)
    predicted   = (predictions > 0.5).float()
    accuracy    = (predicted == y_test).float().mean().item()

print(f"\nTest accuracy: {accuracy: .3f}")
print(f"Correct: {int(accuracy * len(y_test))} / {len(y_test)}")
