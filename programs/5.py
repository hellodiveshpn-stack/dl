import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

# Sample dataset (y = 2x)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Without Regularization
model_no_reg = LinearRegression()
model_no_reg.fit(X, y)

# With L2 Regularization (Ridge)
model_l2 = Ridge(alpha=0.1)   # alpha = lambda
model_l2.fit(X, y)

# Results
print("Without Regularization:")
print("Weight:", model_no_reg.coef_[0])
print("Bias:", model_no_reg.intercept_)

print("\nWith L2 Regularization:")
print("Weight:", model_l2.coef_[0])
print("Bias:", model_l2.intercept_)