import numpy as np
import matplotlib.pyplot as plt
# Generate x once
x = np.linspace(-10, 10, 100)
# -------- Activation Functions -------- #
def sigmoid(x):
return 1 / (1 + np.exp(-x))
def tanh(x):
return np.tanh(x)
def relu(x):
return np.maximum(0, x)
def leaky_relu(x, alpha=0.1):
return np.where(x >= 0, x, alpha * x)
def softmax(x):
e_x = np.exp(x - np.max(x)) # numerical stability
return e_x / np.sum(e_x)
# -------- Generic Plot Function -------- #
def plot_function(x, y, title, ylabel):
plt.plot(x, y)
plt.title(title)
plt.xlabel("Input")
plt.ylabel(ylabel)
plt.grid(True)
plt.show()
# -------- Plot Activations -------- #
plot_function(x, sigmoid(x), "Sigmoid Activation Function", "Sigmoid(x)")
plot_function(x, tanh(x), "Tanh Activation Function", "Tanh(x)")
plot_function(x, relu(x), "ReLU Activation Function", "ReLU(x)")
plot_function(x, leaky_relu(x), "Leaky ReLU Activation Function", "LeakyReLU(x)")
# -------- Softmax Example -------- #
sample = np.array([1, 2, 3])
probabilities = softmax(sample)
plt.bar(["Class A", "Class B", "Class C"], probabilities)
plt.title("Softmax Output")
plt.ylabel("Probability")
plt.show()