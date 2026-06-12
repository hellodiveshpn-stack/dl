import numpy as np

# Input vector (Study score, Attendance score)
X = np.array([0.8, 0.7])

# Hidden layer weights (2 inputs → 3 hidden neurons)
W1 = np.array([[0.5, 0.2, 0.3],
               [0.4, 0.6, 0.1]])
b1 = np.array([0.1, 0.1, 0.1])

# Output layer weights (3 hidden neurons → 3 output classes)
W2 = np.array([[0.7, 0.2, 0.1],
               [0.3, 0.8, 0.2],
               [0.5, 0.4, 0.6]])
b2 = np.array([0.1, 0.1, 0.1])

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Softmax activation
def softmax(x):
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)

# Hidden layer
hidden_input = np.dot(X, W1) + b1
hidden_output = relu(hidden_input)

print("Hidden Layer Output:", hidden_output)

# Output layer
final_input = np.dot(hidden_output, W2) + b2
final_output = softmax(final_input)

print("Class Probabilities:", final_output)

# Prediction
predicted_class = np.argmax(final_output)
print("Predicted Class:", predicted_class)