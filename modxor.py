import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Model configuration
model = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=42
)
model.fit(X, y)

# Create a dense grid
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 500), np.linspace(-0.5, 1.5, 500))

# Predict probabilities
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=50, cmap='bwr', alpha=0.3)
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=100)
plt.title("XOR Decision Boundary (Non-Linear)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.colorbar(label='P(y=1)')
plt.show()