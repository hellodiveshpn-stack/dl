import numpy as np
import matplotlib.pyplot as plt

# Dataset: [Usage Hours, Complaints]
X = np.array([
    [50, 1],
    [80, 0],
    [30, 3],
    [90, 0],
    [20, 4],
    [70, 1]
], dtype=float)

y = np.array([
    [0],
    [0],
    [1],
    [0],
    [1],
    [0]
], dtype=float)

# Normalize input data
X = X / np.max(X, axis=0)

# Network architecture
input_n = 2
hidden_n = 5
output_n = 1

# Initialize weights
np.random.seed(42)
Wxh = np.random.randn(input_n, hidden_n)
Why = np.random.randn(hidden_n, output_n)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training parameters
lr = 0.5
epochs = 10000
cost = []

# Training
for i in range(epochs):

    # Forward propagation
    h_in = np.dot(X, Wxh)
    h_out = sigmoid(h_in)

    o_in = np.dot(h_out, Why)
    y_hat = sigmoid(o_in)

    # Error and MSE
    error = y_hat - y
    mse = 0.5 * np.mean(error ** 2)
    cost.append(mse)

    # Backpropagation
    d_out = error * y_hat * (1 - y_hat)

    d_hidden = np.dot(d_out, Why.T) * h_out * (1 - h_out)

    # Weight update
    Why -= lr * np.dot(h_out.T, d_out)
    Wxh -= lr * np.dot(X.T, d_hidden)

# Plot cost function
plt.plot(cost)
plt.title("Cost Function - Churn Prediction")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Final predictions
print("Predicted Churn Values:")
print(np.round(y_hat, 3))