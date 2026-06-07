import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate dataset
np.random.seed(42)
n=200

df = pd.DataFrame({
    "experience_years": np.random.randint(0,20,n),
    "income": np.random.normal(50000, 15000, n),
    "age": np.random.randint(22, 60, n),
    "hired": (np.random.rand(n) > 0.5)
})

# Scatter plot - experience vs income colored by hired
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    df["experience_years"],
    df["income"],
    c=df["hired"].astype(int),
    alpha =0.6,
    edgecolors="white",
    linewidth=0.5,
    s=60
    )
plt.colorbar(scatter,label="Hired (0=No, 1=Yes)")
ax.set_xlabel("Experience Years")
ax.set_ylabel("Income")
ax.set_title("Experience vs Incom - colored by Hired")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_hired.png", dpi=150)
plt.close()
print("saved scatter-hired.png")

# Histogram - income distribution
fig,ax =plt.subplots(figsize=(8,5))
ax.hist(df["income"], bins=30, color="steelblue", 
        edgecolor="white", alpha=0.8)
ax.axvline(df["income"].mean(), color="red", linestyle="--", 
           linewidth=2, label="mean")
ax.axvline(df["income"].median(), color="green", linestyle="--",
           linewidth=2, label="median")
ax.set_title("Income Distribution")
ax.legend()
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("income_hist.png", dpi=150)
plt.close()
print("save income_hist.png")

# Histogram - age distributio bu hired status
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[df["hired"] == True]["age"], bins = 20,
        color="steelblue", alpha=0.6, label="hired", edgecolor="white")
ax.hist(df[df["hired"] == False]["age"], bins=20,
        color="orange", alpha=0.6, label="not hired", edgecolor="white")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution by Hired Status")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("age_hist.png", dpi=150)
plt.close()
print("save age_hist.png")