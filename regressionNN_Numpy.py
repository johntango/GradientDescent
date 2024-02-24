import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Step 2: Initialize parameters
W = np.random.randn(1, 1)
b = np.zeros(1)

# Step 3: Settings for the learning algorithm
learning_rate = 0.01
n_iterations = 1000

# Step 4: Training process
for i in range(n_iterations):
    # Model prediction
    y_pred = X.dot(W) + b
    
    # Loss calculation
    loss = np.mean((y_pred - y) ** 2) / 2
    
    # Gradient calculation
    W_grad = X.T.dot(y_pred - y) / len(X)
    b_grad = np.sum(y_pred - y) / len(X)
    
    # Parameters update
    W -= learning_rate * W_grad
    b -= learning_rate * b_grad
    
    # Print loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

# Display the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X.dot(W) + b, color='red', label='Linear regression')
plt.legend()
plt.show()

print("Model parameters:")
print(f"Weights: {W.flatten()}")
print(f"Bias: {b}")
