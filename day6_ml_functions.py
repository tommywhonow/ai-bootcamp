import numpy as np

# Softmax - converts scores to probabilities 
def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x)) # subtract max for numerical stability
    return exp_x / np.sum(exp_x)    # type: ignore[no-any-return]

# ReLU - zero out negatives
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0,x)  # type: ignore[no-any-return]

# Mean squared error
def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))

# Linear layer - core of every neural network
def linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b   # type: ignore[no-any-return]

# Test softmax
scores = np.array([2.0, 1.0, 0.5])
probs = softmax(scores)
print("softmax: ")
print("input: ", scores)
print("output", probs)
print("sun: ", np.sum(probs)) # sum is 1

# Test relu
x = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
print("\nrelu:")
print("input:", x)
print("output:", relu(x))

# Test mse
predictions = np.array([2.5, 0.5, 2.0, 8.0])
targets = np.array([3.0, -0.5, 2.0, 7.0])
print("\nmse:")
print("predictions:", predictions)
print("targets:", targets)
print("mse:", mse(predictions, targets))

# Test linear layer
X = np.random.rand(5,3) # 5samples, 3 features
W = np.random.rand(3,4) # 3 imputs, 4 outputs
b = np.zeros(4)         # 4 biasses

output = linear(X, W, b)
print("\nlinear layer:")
print("input shape:", X.shape)
print("weight shape", W.shape)
print("output shape:", output.shape)