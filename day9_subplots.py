import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate data
np.random.seed(42)
n = 200
epochs = np.arange(1,51)

train_loss = 2.0 * np.exp(-0.005 * epochs) + np.random.normal(0, 0.05, 50)
val_loss   = 2.0 * np.exp(-0.004 * epochs) + np.random.normal(0, 0.08, 50)
train_acc  = 1 - np.exp(-.06 * epochs) + np.random.normal(0, 0.02, 50)
val_acc    = 1 - np.exp(-.005 * epochs) + np.random.normal(0, 0.03, 50)

df = pd.DataFrame({
    "income": np.random.normal(50000, 15000, n).round(2),
    "experience": np.random.randint(0, 20, n),
    "hired": (np.random.rand(n) > 0.5)
})

# 2x2 grid subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Training Report", fontsize=16, fontweight="bold")

# Top left - loss curve
axes[0, 0].plot(epochs, train_loss, label="train", color="blue", linewidth=2)
axes[0, 0].plot(epochs, val_loss, label="val", color = "orange",
                linestyle="--", linewidth=2)
axes[0, 0].set_title("Loss Curve")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Top right - accuracy curve
axes[0, 1].plot(epochs, train_acc, label="train", color="green", linewidth=2)
axes[0, 1].plot(epochs, val_acc, label="val", color="lime",
               linestyle="--", linewidth=2)
axes[0, 1].set_title("Accuracy Curve")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Bottom left - income histogram
axes[1, 0].hist(df["income"], bins=30, color="steelblue",
                edgecolor="white", alpha =.08)
axes[1, 0].axvline(df["income"].mean(), color="red", )
axes[1, 0].set_title("Income Distribution")
axes[1, 0].set_xlabel("Income")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

#bottom right - scatter
axes[1, 1].scatter(df["experience"], df["income"],
                   c=df["hired"].astype(int),
                   cmap="coolwarm", alpha=0.6,
                   edgecolors="white", linewidth=0.5)
axes[1, 1].set_title("Experience vs Income")
axes[1, 1].set_xlabel("Experience Years")
axes[1, 1].set_ylabel("Income")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved training_report.png")