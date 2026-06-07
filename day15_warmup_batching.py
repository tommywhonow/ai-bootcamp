import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# TensorDataset wraps tensors
# Use when data is already tensors and you need no custom logic

torch.manual_seed(42)
n = 200
X = torch.randn(n, 1)
y = 2 * X + 1 + 0.1 * torch.randn(n, 1) # learn y = 2x + 1

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]
train_ds = TensorDataset(X_train, y_train)

def train_and_eval(batch_size: int, epochs: int = 100) -> float:
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    for _ in range(epochs):
        for bX, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bX),by)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return float(criterion(model(X_test), y_test).item())

print("Effect of batch size on test loss (lower = better):")
for bs in [1, 8 , 32, 160]:
    loss = train_and_eval(bs)
    label = "full batch" if bs == 160 else f"batch = {bs}"
    print(f"    {label:15s}: test loss = {loss:.4f}")