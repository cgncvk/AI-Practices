import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1) 
y =  4 +3 * X+np.random.randn(100,1)

plt.scatter(X, y)


theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def calc_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate = 0.01, iterations = 100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot(prediction - y))
        theta_history[it, :] = theta.T
        cost_history[it] = calc_cost(theta, X, y)
    return theta, cost_history, theta_history


lr = 0.01
n_iter = 1000
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)
