import torch
import numpy as np

print("=== Tensor Creation ===")
t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
print (f"From list: {t1}")
print(f"dtype:      {t1.dtype}")
print(f"shape:      {t1.shape}\n")

print("=== Common Operations===")
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"a + b = \n{a + b}")
print(f"a @ b = \n{a @ b}")
print(f"a.T = \n{a.T}\n")

print("=== Shapes===")
x = torch.randn(3, 4)
print(f"shape:    {x.shape}")
print(f"numel:    {x.numel()}")
print(f"reshape: {x.reshape(2,6).shape}")
print(f"flatten:  {x.reshape(-1).shape}\n")

print("==== NumPy Bridge ===")
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)
print(f"numpy -> tensor: {t}")
print(f"tensor -> numpy: {t.detach().numpy()}\n")

print("=== Device ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
print(f"tensor on {device}: {x.shape}")
