import cvxpy as cp
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# L1 norm of the parameter vector.
def l1_norm(beta):
    return cp.norm1(beta)

# Mean squared error.
def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_function(X, Y, beta).value

# Loss function.
def loss_function(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

# Objective function to minimize.
def objective_function(X, Y, beta, lambd):
    return loss_function(X, Y, beta) + lambd * l1_norm(beta)

# Generate the data set.
def generate_data(n_samples=100, n_features=20, sigma=5, density=0.2):
    "Generates normally distributed data."
    bias = 10
    # To be able to replicate simulations.
    np.random.seed(1)
    # Initialising the feature vector.
    beta_star = np.random.randn(n_features)
    # Randomly choose and set 1-density*n_features features to null.
    idxs = np.random.choice(range(n), int((1-density)*n_features), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    # Generate random input.
    X = np.random.randn(n_samples,n_features)
    # Generate observations.
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=n_samples)
    return X, Y, beta_star

m = 1000
n = 40
sigma = 5
density = 0.1

X, Y, coef = generate_data(m, n, sigma, density)

X_train = X[:int(m/2), :]
Y_train = Y[:int(m/2)]
X_test = X[int(m/2):, :]
Y_test = Y[int(m/2):]

# Set the decision variable.
beta = cp.Variable(n)
# Set a constraint on the hyper-parameter.
lambd = cp.Parameter(nonneg=True)
# Define the problem.
problem = cp.Problem(cp.Minimize(objective_function(X_train, Y_train, beta, lambd)))

# Define an interval of lambda values to be tested.
lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    # Set the hyper-parameter.
    lambd.value = v
    # Solve the objective function for a specific lambda.
    problem.solve(verbose=True)
    # Calculate the error (MSE).
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)

# Plot how the MSE depends on the hyper-parameter.
def plot_train_test_errors(train_errors, test_errors, lambd_values, title):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title(title)
    plt.show()

# Plot the parameters approaching zero.
def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()


plot_train_test_errors(train_errors, test_errors, lambd_values, "Mean Squared Error (MSE)")
plot_regularization_path(lambd_values, beta_values)