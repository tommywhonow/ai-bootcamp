import numpy as np
from sklearn.model_selection import train_test_split

# ACTIVATION FUNCTIONS
def relu(Z: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.maximum(0, Z)
    return result

def relu_grad(Z: np.ndarray) -> np.ndarray:
    # Derivative of Relu
    # Used during backpropagation
    return (Z > 0).astype(float)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1 / (1 + np.exp(-Z))
    return result

def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))

# CREATE DATA
np.random.seed(42)
n          =  500
age        = np.random.randint(22, 60, n).astype(float)
income     = np.random.randint(20000, 100000, n).astype(float)
education  = np.random.randint(8, 22, n).astype(float)
experience = np.random.randint(0, 20, n).astype(float)
hired = (0.3 * education + 0.5 * experience
         + np.random.normal(0, 2, n) > 10).astype(float)

X = np.column_stack([age, income, education, experience])
y = hired.reshape(-1, 1)

# NORMALISE - neural networks train better when inputs are on similar scales
# (X - mean) / std transforms each feature to mean=0, std=1
X = (X- X.mean(axis=0)) / X.std(axis=0)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# INITIALISE WEIGHTS
np.random.seed(42)
W1 = np.random.randn(4, 8) * np.sqrt(2.0 / 4)
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 4) * np.sqrt(2.0 / 4)
b2 = np.zeros((1, 4))
W3 = np.random.randn(4, 1) * np.sqrt(2.0 / 4)
b3 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 1000

# TRAINING LOOP
print("Training...")
for epoch in range(epochs):

    # FORWARD PASS
    Z1 = X_train @ W1 + b1  #(400, 8)
    A1 = relu(Z1)           #(400, 8)
    Z2 = A1 @ W2 + b2       #(400, 4)
    A2 = relu(Z2)           #(400, 4)
    Z3 = A2 @ W3 + b3       #(400, 1)
    A3 = sigmoid(Z3)        #(400, 1) predictions

    loss = bce(y_train, A3)

    #BACKWARD PASS
    n_train = X_train.shape[0]

    dZ3 = A3 - y_train
    dW3 = A2.T @ dZ3 / n_train
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_grad(Z2)
    dW2 = A1.T @ dZ2 / n_train
    db2 = np.mean(dZ2, axis=0, keepdims = True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X_train.T @ dZ1 / n_train
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 100 == 0:
        print(f"    Epoch {epoch+1: 4d} | Loss: {loss:.4f}")

# EVALUATE
Z1t = X_test @W1 + b1
A1t = relu(Z1t)
Z2t = A1t @ W2 + b2
A2t = relu(Z2t)    
Z3t = A2t @ W3 +b3
A3t = sigmoid(Z3t)
y_pred = (A3t > 0.5).astype(int)
accuracy = float(np.mean(y_pred == y_test))
print(f"\nTest accuracy: {accuracy:.3f}")
print(f"Correct: {int(accuracy * len(y_test))} / {len(y_test)}")