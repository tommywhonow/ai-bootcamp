import pandas as pd
import numpy as np

# Generate a realistic ML dataset
np.random.seed(42)
n = 200

data: dict[str, list[object]] = {
    "age": np.random.randint(18, 65, n).tolist(),
    # min = 16, max = 65(exclude), n datas, 
    # 3.tolist() converts numpy array to python list. pandas use list
    "income": np.random.normal(50000, 15000, n).round(2).tolist(),
    # generate from a normal distribution: mean (50000) std(15000), .round(2) 2 decimals
    "education_years": np.random.randint(8, 22, n).tolist(),
    "experience_years": np.random.randint(0, 40, n).tolist(),
    "hired": (np.random.rand(n) > 0.5).tolist()
    # random.rand() generates 0-1. (np.random.rand(n) > 0.5) generates boolean
}

df = pd.DataFrame(data)

# Basic exploration
print("=== SHAPE ===")
print(df.shape)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n === DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== STATISTICS ===")
print(df.describe)

print("\n=== HIRED DISTRIBUTION ===")
print(df["hired"].value_counts())

print("\n=== AVERAGE INCOME BY HIRED ===")
print(df.groupby("hired")["income"].mean())

print("\n === CORRELATION===")
print(df.corr(numeric_only=True).round(2))

# Featured Engineering - create new columns
df["income_per_exp"] = df["income"] / (df["experience_years"] +1 )
df["senior"] = df["age"] > 40

# Filter and analyse
print(df[["age", "hired"]].head(20))
senior_hired = df[(df["senior"] == True) & (df["hired"] == True)]
print(f"\nsenior hired: {len(senior_hired)} out of {len(df[df['age'] > 40].value_counts())} seniors")