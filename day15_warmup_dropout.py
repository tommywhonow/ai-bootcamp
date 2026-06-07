import torch
import torch.nn as nn

print("=== Dropout Behaviour ===")
dropout = nn.Dropout(p=0.5) # 50% of values is set to 0
x = torch.ones(1, 10)
print(f"Input {x}")

# Training mode - dropout active
dropout.train()
print(f"Train run 1: {dropout (x)}")
print(f"Train run 2: {dropout (x)}")
print(f"Train run 3: {dropout (x)}")
print("Notice: different neurons zeroed each run\n")

# Eval mode - dropout disabled
dropout.eval()
print(f"Eval mode:  {dropout(x)}")
print("Notice: all values pass through unchanged\n")

# Scale Compensation
print("=== Scale Compensation ===")
# During training, remaining neurons scaled up by 1/(1-p)
# With p = 0.5: survivors multiplied by 2.0
# This keeps expected value the same in ttrain and eval
dropout.train()
runs = torch.stack([dropout(torch.ones(1, 1000)) for _ in range(100)])
print(f"Mean of 100 training runs: {runs.mean().item():.3f} (should be -1.0)")
