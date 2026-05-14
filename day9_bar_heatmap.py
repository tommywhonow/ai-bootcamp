import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate dataset
np.random.seed(42)
n=200

df = pd.DataFrame({
    "age": np.random.randint(22, 60, n),
    "income": np.random.normal(50000, 15000, n).round(2),
    "education_years": np.random.randint(8, 22, n),
    "experience_years": np.random.randint(0, 20, n),
    "hired": (np.random.rand(n)>0.5).astype(int)
})

# Bar Chart - hired vs not hired count
fig, ax = plt.subplots(figsize=(7,5))
counts = df["hired"].value_counts().sort_index()
bars = ax.bar(["Not Hired", "Hired"], counts.tolist(),
              color=["orange", "steelblue"],
              edgecolor="white", width=0.5)

for bar, val in zip(bars, counts.values): 
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            str(val), ha="center", fontsize=11)

ax.set_ylabel("Count")
ax.set_title("Hired vs Not Hired")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("bar_hired.png", dpi=150)
plt.close()
print("save bar_hired.png")

# Heatmap - correlation matrix
corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
            ha="center", va="center", fontsize=9)

ax.set_title("correlation Matrix")
plt.tight_layout()
plt.savefig("heatmap_corr.png", dpi=150)
plt.close()
print("saved heatmap_corr.png")