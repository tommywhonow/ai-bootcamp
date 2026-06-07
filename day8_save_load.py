import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n = 100

data: dict[str, list[object]] = {
    "name": [f"person_{i}" for i in range(n)],
    "age": np.random.randint(18, 65, n).tolist(),
    "income": np.random.normal(50000, 15000, n).round(2).tolist(),
    "hired": (np.random.rand(n) > 0.5).tolist()
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data.csv", index=False)
print("save to data.csv")

# Load from CSV
df_loaded = pd.read_csv("data.csv")
print("loaded shape: ", df_loaded.shape)
print("\nfirst three rows:\n", df_loaded.head(3))

# Save to JSON
df.to_json("data.json", orient="records", indent=2)
print("\nsaved to data.json")

# Load from JSON
df_json = pd.read_json("data.json")
print("load from json shape", df_json.shape)

# Saveonly clean data
df_clean = df[df["hired"] == True].copy().reset_index(drop=True)
df_clean.to_csv("hired_only.csv", index=False)
print(f"\nsaved{len(df_clean)} hired peopel to hired_only.csv")

# Verify
df_verify = pd.read_csv("hired_only.csv")
print("verified shape: ", df_verify.shape)
print("all hired: ", df_verify["hired"].all())
