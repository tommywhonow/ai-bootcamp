import numpy as np

# Creating arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3,3))
c = np.ones((2,4))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)

print("array: ", a)
print("zeros:\n", b)
print("ones:\n", c)
print("arange: ", d)
print("linspace: ", e)

# Shape and dtype
print("\nshape: ", a.shape)
print("dtype: ", a.dtype)
print("ndim:", a.ndim)

# Reshaping
matrix = np.arange(12).reshape(3,4)
print("\nshape:\n", matrix)
print("shape:", matrix.shape)

# Indexing
print("\nfirst row:", matrix[0])
print("element [1,2]:", matrix[1,2])
print("first column:", matrix[:,0])
print("submatrix:\n", matrix[0:2, 1:3])
