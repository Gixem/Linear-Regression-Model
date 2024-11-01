## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


# Read data matrix X and labels y from text file.
def read_data(file_name):
    data = np.loadtxt(file_name)
    X = data[:, 0]  # İlk sütun (floor size)
    y = data[:, 1]  # İkinci sütun (price)
    return X, y



# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lr, epochs):
    w = 0
    b = 0
    m = len(y)
    costs = []

    for epoch in range(epochs):
        y_pred = X * w + b
        error = y_pred - y
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        costs.append(cost)

        # Gradient calculation
        w_grad = (1 / m) * np.dot(X.T, error)
        b_grad = (1 / m) * np.sum(error)

        # Update parameters
        w -= lr * w_grad
        b -= lr * b_grad

    return w, b, costs


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w, b):
    y_pred = X * w + b
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return rmse


# Compute objective (cost) function.
def compute_cost(X, y, w, b):
    m = len(y)
    y_pred = X * w + b
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
    m = len(y)
    predictions = X @ w
    grad = (1/m) * (X.T @ (predictions - y))  # Gradyan hesaplama
    return grad



##======================= Main program =======================##

# Read the training and test data.
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

# Standardize training data
mean, std = scaling.mean_std(Xtrain)
Xtrain = scaling.standardize(Xtrain, mean, std)
Xtest = scaling.standardize(Xtest, mean, std)

# Train the model
lr = 0.1
epochs = 500
w, b, costs = train(Xtrain, ttrain, lr, epochs)

# Compute RMSE for train and test
train_rmse = compute_rmse(Xtrain, ttrain, w, b)
test_rmse = compute_rmse(Xtest, ttest, w, b)
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Plot cost vs. epochs
plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost vs. Epochs')
plt.show()

# Plot training and test data with the linear regression line
plt.scatter(Xtrain, ttrain, color='blue', label='Train data')
plt.scatter(Xtest, ttest, color='green', label='Test data')
x_values = np.linspace(min(Xtrain), max(Xtrain), 100)
y_values = w * x_values + b
plt.plot(x_values, y_values, color='red', label='Regression line')
plt.xlabel('Floor size')
plt.ylabel('Price')
plt.legend()
plt.show()
