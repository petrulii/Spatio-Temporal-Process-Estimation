import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from torch import arctan

def arctangent(x):
  return np.arctan(x)

def neg_inverse(x):
  return -1/x

def soft_plus(x):
  return np.log(1+np.exp(x))

def gelu(x):
  return x/2*(1+erf(x/np.sqrt(2)))

# Inverse of the link function.
def activation(lc):
  return np.array(list(map(arctangent, lc)))

# Loss function.
def loss(X, Y, b):
  return np.sum((Y - activation(b*X))**2) / float(len(X))

# Generating some data.
np.random.seed(1)
n_features = 1
n_samples = 1000

# Plotting the loss function in terms of beta.
generated_beta = np.linspace(-1000, 1000, n_samples)
true_beta = np.random.randn(n_features)
X = np.array([np.random.randn(n_features) for i in range(n_samples)])
Y = activation(np.multiply(X,[true_beta] * n_samples))# + np.random.randn(n_samples)
plt.plot(generated_beta, [loss(X,Y,b) for b in generated_beta], label="Loss function")
plt.legend(loc="upper left")
plt.xlabel(r"$\beta$")
plt.ylabel("loss")
plt.show()