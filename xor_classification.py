import numpy as np
import matplotlib.pyplot as plt

# Plot XOR Points
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)

# Draw a falling line
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = -x_vals + 0.5 # Example line: y = -x + 0.5
plt.plot(x_vals, y_vals, 'k--', label = "Linear Boundary")
plt.legend()

# Train a tiny neural network
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(2,), # 2 neurons in 1 hidden layer
    activation='tanh', # Non-linear activation 
    solver='adam', # Better optimizer for small datasets
    max_iter=10000, # Increase iterations 
    tol=1e-8, # Lower tolerance for convergence 
    random_state=42 # Reproducibility
)
model.fit(x,y)

# Print training diagnostics
print(f"Training iterations:{model.n_iter_}")
print(f"Training accuracy: {model.score(x, y)}")

"""Plot the non-linear Boundary"""
# Create a Grid
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 500), np.linspace(-0.5, 1.5, 500))
z = model.predict(np.c_[xx.ravel(), yy.ravel()]). reshape(xx.shape)

# Plot Contours
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z, alpha=0.3, cmap='bwr', levels=[0, 0.5, 1])
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', edgecolors='k', s=100)
plt.title("XOR Solved by Neural Network ( Non-Linear Boundary)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()

print("Predictions:", model.predict(x)) # output should be [0, 1, 1, 0]
# Make it interactive
from ipywidgets import interact, FloatSlider

def plot_boundary(m=1, b=0):
    plt.figure(figsize=(6,6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
    y_vals = m * x_vals + b
    plt.plot(x_vals, y_vals, 'k--')
    plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
    plt.title(f"y = {m:.1f}x + {b:.1f}")

interact(plot_boundary, m =FloatSlider(min = -2, max =2, step=0.1, value=1),
                        b = FloatSlider(min= -1, max = 1, step=0.1, value=0))
