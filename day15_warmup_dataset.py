import torch
from torch.utils.data import Dataset, DataLoader

class NumbersDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    # X = numbers 0- --99, y=1 ifeven, 0 if odd
    def __init__(self, size: int = 100) -> None:
        self.X = torch.arange(0, size, dtype=torch.float32).reshape(-1, 1)
        self.y = (self.X % 2 == 0).float()
    
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

dataset =  NumbersDataset(100)
print(f"Dataset size: {len(dataset)}")
x0, y0 = dataset[0]
x1, y1 = dataset[1]
print(f"Sample 0: X={x0.item}, y={y0.item()} (0 is even -> 1.0)")
print(f"Sample 1: X={x1.item()}, y={y1.item()} (1 is odd -> 0.0) \n")

# Wrap in Dataloader
loader = DataLoader(dataset, batch_size=10, shuffle=False)
print("First 3 batches:")
for i, (bX, by) in enumerate(loader):
    print(f"    Batch {i+1}: X shape = {bX.shape}, y shape={by.shape}")
    print(f"    X values: {bX.squeeze().tolist()}")
    if i >= 2:
        break

print(f"\nTotal batches: {len(loader)} (expected; 10)")

# Shuffle loader
loader_s = DataLoader(dataset, batch_size=10, shuffle=True)
print("shuffled first batch:")
for bX, by in loader_s:
    print(f"    {bX.squeeze().tolist()}")
    break