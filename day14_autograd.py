import torch

print("=== Basic Autograd ===")
x = torch.tensor(3.0, requires_grad=True)
y = x **2 + 2 * x + 1
y.backward() # type: ignore[no-untyped-call]
print(f"x =  {x.item()}")
print(f"y = x^2 + 2x + 1 = {y.item()}") 
assert x.grad is not None
print(f"dy/dx = 2x + 2 = {x.grad.item()} (expected: 8.0)\n")

print("=== Vector Autograd ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
y = (x ** 2).sum() # sum so we can call backward()
y.backward() # type: ignore[no-untyped-call]
print(f"x     = {x.data}")
print(f"dy/dx = 2x = {x.grad} (expected: [2, 4, 6]\n")

print("=== Gradient Accumulation Warning ===")
x = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    y =  x * 2
    y.backward()# type: ignore[no-untyped-call]
    assert x.grad is not None
    print(f"    Iteration {i+1}: x.grad = {x.grad.item()} accumulates!")

# Reset and do it correctly
x = torch.tensor(2.0, requires_grad=True)
print("Correct way:")
for i in range(3):
    if x.grad is not None:
        x.grad.zero_() # zero gradient before each backward
    y = x * 2
    y.backward() # type: ignore[no-untyped-call]
    assert x.grad is not None
    print(f"    Iteration {i+1} x.grad = {x.grad.item()} (always 2.0)\n")

print("=== no_grad context")
x =  torch.randn(3, 3, requires_grad=True)
with torch.no_grad():
    y = x * 2
    print(f"requires_grad inside no_grad: {y.requires_grad}") #False