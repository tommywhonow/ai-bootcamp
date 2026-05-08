import numpy as np

# scalar broadcasting
a = np.array([1, 2, 3, 4, 5])
print("a: ", a,  "\n")
print("a+10: ", a+10, "\n")
print("a*3: ", a*3, "\n")

# Vector broadcast across matrix
matrix = np.ones((3,4))
vector = np.array([1, 2, 3, 4])
print("\nmatrix:\n", matrix)
print("vector: ", vector)
print("matrix + vector:\n", matrix + vector)

# Normalisation - subtract mean from every row
X = np.array([
    [85, 92, 78],
    [70, 95, 60],
    [90, 88, 95]
], dtype=float)

mean = X.mean(axis=0)
std = X.std(axis=0)

print("\nX:\n", X)
print("mean: ", mean)
print("std: ", std)

X_normalised = (X - mean)/std
print(" normalised:\n", X_normalised)

# axis explained
print("\nsum axis=0 (down columns): ", X.sum(axis=0))
print("sum axis=1 (across rows:)", X.sum(axis=1))
print("sum all:", X.sum())