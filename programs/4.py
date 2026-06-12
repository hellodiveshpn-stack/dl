import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. Generate Random Dataset
np.random.seed(42)
n_samples = 200

# Generate 6 random medical-like features
glucose = np.random.uniform(70, 200, n_samples)
blood_pressure = np.random.uniform(50, 120, n_samples)
bmi = np.random.uniform(18, 45, n_samples)
age = np.random.uniform(18, 70, n_samples)
insulin = np.random.uniform(0, 300, n_samples)
pregnancies = np.random.randint(0, 10, n_samples)

# Combine features into matrix
X = np.column_stack((
    glucose,
    blood_pressure,
    bmi,
    age,
    insulin,
    pregnancies
))

# Create target (rule-based for realism)
y = ((glucose > 140) & (bmi > 30)).astype(int)

# 2. Normalize Data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 3. Build ANN (Two-Layer)
model = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.3,
    max_iter=300,
    random_state=1
)

# 4. Train Model
model.fit(X, y)

# 5. Plot Cost Function
plt.figure()
plt.plot(model.loss_curve_)
plt.title("Cost Function (MSE) vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# 6. Final Predictions
predictions = model.predict(X)

binary_predictions = (predictions > 0.5).astype(int)

print("\nFirst 10 Original Targets:")
print(y[:10])

print("\nFirst 10 Raw Predictions:")
print(predictions[:10])

print("\nFirst 10 Final Binary Predictions (0=Non-Diabetic, 1=Diabetic):")
print(binary_predictions[:10])

# 7. Compare Original vs Predicted
print("\nComparison (Original vs Predicted):")
print("Original | Predicted")

for i in range(10):
    print(f"{y[i]} | {binary_predictions[i]}")

# 8. Accuracy
accuracy = accuracy_score(y, binary_predictions)

print("\nTest Accuracy:", accuracy)