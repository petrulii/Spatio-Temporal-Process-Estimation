import cvxpy as cp
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def l2_norm(z):
    """ Calculates the L2 norm of a given vector z. """
    return 1/2 * cp.norm2(z)**2

def projection(y, n):
    """ Calculates the projection from point y onto a point x in the probability simplex. """
    # Set the decision variable.
    x = cp.Variable(n)
    # Define the problem.
    unit = np.ones(n)
    problem = cp.Problem(cp.Minimize(l2_norm(y-x)), [x @ unit == 1, x >= 0])
    # Solve.
    problem.solve()
    return x.value

def theoretical_stepsize(H):
    """ Calculates the theoretical optimal step size based on the Hessian of size n*m of the bilinear objective function. """
    U, s, VT = linalg.svd(H)
    return 1/np.sqrt(np.max(s))

def linesearch_stepsize(A, x, y, x_new, x_grad, rate):
    """ Calculates the a step size based on a condition. """
    beta = np.sqrt(2)
    while ((np.array(rate*x_grad)).T.dot(y-x_new) - np.power(linalg.norm(x-x_new, ord=2),2) <= 0):
        rate_prev = rate
        rate *= beta
    return rate_prev

def gradient(A, x, y):
    return A.dot(y), -np.transpose(A).dot(x)

def gradient_descent(A, x_init, y_init, rate, fig, ax, index = 0, iterations = 200):
    x, y = x_init, y_init
    x_values, y_values = [], []
    L2_norm, L1_norm = [], []

    for i in range(iterations):
        x_grad, y_grad = gradient(A, x, y)
        x = x - rate*(x_grad)
        x_values.append(x[index])
        y = y - rate*(y_grad)
        y_values.append(y[index])
        #L2_norm.append(x**2 + (x_values[i-1])**2)
        #L1_norm.append(np.abs(x-x_values[i-1]))

    ax.plot(x_values, y_values, alpha=1,linewidth=0, label='gamma=0.5$', marker='x', color='b')
    ax.plot(x_values[0], y_values[0], alpha=1, linewidth=2, label='gamma=0.5$', marker='o', color='r')
    #plt.plot(L1_norm, label='gamma=0.5$')
    #fig.show()

def extragradient(A, n, x_init, y_init, rate, fig, ax, index = 0, max_iterations = 1500, adaptive = False):
    x, y = x_init, y_init
    x_values, y_values = [], []
    norm, L1_norm = [], []
    x_grad_ = 0
    i = 0

    while True:
        # Gradient step to go to an intermediate point.
        x_prev, y_prev = x, y
        x_grad, y_grad = gradient(A, x, y)
        x_ = projection(x - rate*(x_grad), n)
        y_ = projection(y - rate*(y_grad), n)

        # Use the gradient of the intermediate point to perform a gradient step from the
        x_grad_prev = x_grad_
        x_grad_, y_grad_ = gradient(A, x_, y_)
        x = projection(x - rate*(x_grad_), n)
        y = projection(y - rate*(y_grad_), n)

        x_values.append(x[index])
        y_values.append(y[index])
        if (linalg.norm(x_prev-x, ord=2) <= 0.0001 and linalg.norm(y_prev-y, ord=2) <= 0.0001) or i >= max_iterations:
            break
        if adaptive == True:
            rate = linesearch_stepsize(A, x_prev, x_, x, x_grad_prev, rate)
        i+=1

    ax.plot(x_values, y_values, alpha=1,linewidth=0, label='gamma=0.5$', marker='x', color='g')
    ax.plot(x_values[0], y_values[0], alpha=1, linewidth=2, label='gamma=0.5$', marker='o', color='r')
    if adaptive == True: fig.savefig("adaptive_extragradient")
    else: fig.savefig("extragradient")
    return x, y, i