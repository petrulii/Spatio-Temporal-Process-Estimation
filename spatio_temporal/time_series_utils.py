import numpy as np

# Identity activation function.
def identity(x):
    return x

# Linear binary activation function.
def binary_linear(x):
    if x >= 0.5 : return 1
    else : return 0

# Mean squared error.
def MSE(series, theta, N, L, d):
    """ Gradient of the negative logistic loss. """
    total_err = 0
    for s in range(d,N):
        X = series[(s-d):s].flatten()
        for l in range(L):
            y = series[s][l]
            y_pred = np.dot(X, theta[l])
            # MSE loss.
            total_err += (np.sum((y - y_pred)**2)).mean()
    return total_err/N

# Log-it activation function.
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Binary log-it activation function.
def binary_sigmoid(x):
    if 1/(1+np.exp(-x)) >= 0.5: return 1
    else: return 0

# Logistic loss gradient.
def log_loss_gradient(X, y, y_pred, l, L, d):
    """ Gradient of the parameters in respect to negative log-loss. """
    return np.dot(X, (y_pred[l] - y[l])) / (d*L)

# Logistic loss gradient.
def log_loss(series, theta, N, L, d):
    """ Gradient of the negative logistic loss. """
    total_err = 0
    for s in range(d,N):
        X = series[(s-d):s].flatten()
        for l in range(L):
            y = series[s][l]
            y_pred = sigmoid(np.dot(X, theta[l]))
            # Logistic loss.
            total_err += (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()
    return total_err/N

def estimate_parameters(series, N, L, d, rate = 0.05, max_error = 0.2, max_iterations = 300, activation = sigmoid, loss = log_loss, gradient = log_loss_gradient):
    """ Gradient descent for time series of 2-D Bernouilli events.
    param series : a time series of 2-D categorical events
    param N : lenght of the time series
    param L : number of locations in the 2-D grid where categorical events take place
    param d : memory depth describing the depth of the autoregression
    param rate : learning rate for gradient descent
    param max_error : maximum tolerated error
    param max_iterations : maximum tolerated number of gradient steps
    """
    theta = np.zeros(shape=(L,d*L))
    theta_grad = np.zeros(shape=(L,d*L))
    y = np.ones(L)
    y_pred = np.zeros(L)
    error = [loss(series, theta, N, L, d)]
    i = 0

    while i < max_iterations and error[i] >= max_error:
        # For each time instance in the time horizon from d to N.
        for s in range(d,N):
            # Take values from the last d time instances. 
            X = series[(s-d):s].flatten()
            y = series[s]
            # For each location in the 2D grid of the current time instance.
            for l in range(L):
                # Predict the value of the current time instance.
                y_pred[l] = activation(np.dot(X, theta[l]))
                # Update the parameter vector.
                theta_grad = gradient(X, y, y_pred, l, L, d)
                theta[l] = theta[l] - rate * theta_grad
        # Calculate the prediction error over the time horizon.
        error.append(loss(series, theta, N, L, d))
        i+=1

    return theta, i, error