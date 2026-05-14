import matplotlib.pyplot as plt
import numpy as np

# Simulate training data
np.random.seed(42)
epochs = np.arange(1, 51)

train_loss = 2.0 * np.exp(-.05 * epochs) + np.random.normal(0, 0.05, 50)
val_loss = 2.0 * np.exp(-.04 * epochs)+ np.random.normal(0, 0.08, 50)
train_acc = 1 - np.exp(-.06 * epochs) + np.random.normal(0, 0.02, 50)
val_cc = 1 - np.exp(-.05 * epochs) + np.random.normal(0, 0.03, 50)

# Loss curve
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(epochs, train_loss, label="train loss", color="blue", linewidth=2)
ax.plot(epochs, val_loss, label="val loss", color ="orange")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()
print("save loss_curve.png")

# Accuracy Curve
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(epochs, train_acc, label="train acc", color="green", linewidth=2)
ax.plot(epochs, val_cc, label="val acc", color="lime", linestyle="--", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("accuracy")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.close()
print("save accuracy_curve.png")