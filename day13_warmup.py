import numpy as np

# Exercise 1: ReLU
def relu(Z: np.ndarray) -> np.ndarray: 
    result: np.ndarray = np.maximum(0,Z)
    return result

test = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
print("Ex 1 - ReLU:")
print(f" input: {test}")
print(f" output: {relu(test)}")
print(f" Expected:[0. 0. 0. 2. 5. ]\nm")

# Exercise 2: Sigmoid
def sigmoid(Z: np.ndarray) -> np.ndarray: 
    result: np.ndarray = 1 / (1 + np.exp(-Z))
    return result
test2 = np.array([-2.0, 0.0, 2.0])
print("Ex 2 - sigmoid:")
print(f" Input: {test2}")
print(f" Output: {sigmoid(test2).round(3)}")
print(f" Expected: [0.119 0.5  0.881]\n")

# Exercise 3: Binary Cross-Entropy
def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-18, 1-1e-8)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_perfect = np.array([0.999, 0.001, 0.999, 0.001])
y_random = np.array([0.5, 0.5, 0.5, 0.5])
print("Ex 3 - Binary Cross-Entropy:")
print(f" Perfect loss: {bce(y_true, y_perfect):.4f} (near 0)")
print(f" Random loss: {bce(y_true, y_random):.4f} (near0.693)\n")

# Exercise 4: One forward pass
np.random.seed(42)
X_s = np.random.randn(5, 4)
W_s = np.random.randn(4, 8) * 0.01
b_s = np.zeros((1, 8))
print("Ex 4 - Forward Pass:")
Z_s =  X_s @ W_s + b_s
A_s = relu(Z_s)
print(f" Output shape: {A_s.shape} (should be (5, 8))")
print(f" All >= 0: {(A_s >= 0).all()}\n")

# Exercise 5 Weight initialisation
np.random.seed(42)
W1 = np.random.randn(4, 8) * 0.01
b1 = np.zeros((1, 8))
W2 = np.random.randn(8,4) * 0.01
b2 = np.zeros((1, 4))
W3 = np.random.randn(4,1) * 0.01
b3 = np.zeros((1, 1))
print("Ex 5 - Weight Initialisation:")
print(f" W1: {W1.shape} W2: {W2.shape} W3: {W3.shape}")
total = W1.size+b1.size+W2.size+b2.size+W3.size+b3.size
print(f" Total params: {total}\n")

# Exercise 6 - Gradient descent step
W_b = np.array([[0.5, -0.3], [0.2, 0.8]])
dW = np.array([[0.1, 0.2], [0.3, 0.1]])
lr = 0.1
W_a = W_b - lr * dW
print("Ex6 - Gradient Descent:")
print(f" Before: {W_b}")
print(f" After: {W_a}")
