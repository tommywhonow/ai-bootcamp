import numpy as np
import time
# vectorisation vs loops
size = 1_000_000
a = np.random.rand(size)
b = np.random.rand(size)

# Python loop - slow
start = time.time()
result_loop = [a[i] + b[i] for i in range(size)]
end = time.time()
print(f"loop: {end - start:.4f} seconds")

# NumPy vecotorised - fast
start = time.time()
result_numpy = a + b
end = time.time()
print(f"numpy: {end - start: .4f} seconds")

#math operations - no loops needed
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("\nx: ", x)
print("x * 2: ", x * 2)
print("x ** 2:", x ** 2)
print("sqrt(x):", np.sqrt(x))
print("sum:", np.sum(x))
print("mean:", np.mean(x))
print("std:", np.std(x))

# Matrix multiplication
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])

print("\nA:\n", A)
print("B:\n", B)
print("A @ B:\n", A @ B)        # matrix multiply
print("A * B:\n", A * B)        # element-wise multiply


