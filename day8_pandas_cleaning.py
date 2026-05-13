import pandas as pd
import numpy as np

# Dataset with real-world messiness
data: dict[str, list[object]] = {
    "name": ["Alice", "Bob", None, "Dave", "Eve", "Bob"],
    "age": [24, 27, 22, None, 25, 27],
    "score": [88.5, 72.0, 95.5, 61.0, None, 72.0],
    "grade": ["A", "B", "A", "C", "B", "B"]
}

df = pd.DataFrame(data)
print("original:\n", df)

# Check missing values
print("\nmissing values:\n", df.isnull().sum())

# Drop rows with missing values
df_filled = df.copy()
df_filled["age"] = df_filled["age"].fillna(df_filled["age"].median())
df_filled["score"] = df_filled["score"].fillna(df_filled["score"].mean())
df_filled["name"] = df_filled["name"].fillna("unknown")
print("\nafter drop_duplicaes:\n", df_filled)

# Remove duplicates
df_clean = df_filled.drop_duplicates()
print("\nafter drop_duplicates:\n", df_clean)

# Add new column
df_clean = df_clean.copy()
df_clean["passed"] = df_clean["score"] > 75
print("\nwith passed column:\n", df_clean)

# Group by and aggregate
print("\nagerage score by grade:\n",
        df_clean.groupby("grade")["score"].mean())

# Sort
print("\nsorted by score:\n",
      df_clean.sort_values("score", ascending=False))