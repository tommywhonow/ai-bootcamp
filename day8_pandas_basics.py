import pandas as pd
import numpy as np

# Create a dataframe from scratch
data: dict[str, list[object]] ={
    "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
    "age": [24, 27, 22, 31, 25],
    "score": [88.5, 72.0, 95.5, 61.0, 83.5],
    "passed": [True, False, True, False, True]
}

df = pd.DataFrame(data)

#Explore
print("shape: ", df.shape)
print("\n Head:\n", df.head())
print("\ndtypes:\n", df.dtypes)
print("\ndecribe:\n", df.describe())

# Selecting
print("\nname column:\n", df["name"])
print("\nname and score:\n", df[["name", "score"]])
print("\nfirst row:\n", df.iloc[0])

# Filtering
print("\npassed students:\n0", df[df["passed"] == True])
print("\nhigh scores:\n", df[df["score"] > 80])