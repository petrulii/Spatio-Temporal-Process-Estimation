import cvxpy as cp
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def l2_norm(z):
    """ Calculates the L2 norm of a given vector z. """
    return 1/2 * cp.norm2(z)**2

def projection(y, d):
    """ Calculates the projection from point y onto a point x in the probability simplex. """
    # Set the decision variable.
    x = cp.Variable(d)
    # Define the problem.
    unit = np.ones(d)
    problem = cp.Problem(cp.Minimize(l2_norm(y-x)), [x @ unit == 1, x >= 0])
    # Solve.
    problem.solve()
    return x.value

def projection_simplex_sort(y, d, z=1):
    """ Calculates the projection from point y onto a point x in the probability simplex. """
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(d) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(y - theta, 0)
    return w

def theoretical_stepsize(H):
    """ Calculates the theoretical optimal step size based on the Hessian of size n*m of the bilinear objective function. """
    U, s, VT = linalg.svd(H)
    return 1/np.sqrt(np.max(s))

def linesearch_stepsize(A, x_i, y_i, x_i_1, grad_y_i, x2_i, y2_i, x2_i_1, grad_y2_i, rate):#(A, x_i, y_i, x_i_1, grad_y_i, rate):
    """ Backtracking line search for step size. """
    i=0
    while i<2:
        if (rate*np.dot((grad_y_i.T),(y_i-x_i_1)) <= (1/2)*np.power(linalg.norm(x_i-x_i_1, ord=2),2)
        and rate*np.dot((grad_y2_i.T),(y2_i-x2_i_1)) <= (1/2)*np.power(linalg.norm(x2_i-x2_i_1, ord=2),2)):
            beta = np.sqrt(2)
        else:
            beta = 0.5
        rate *= beta
        i+=1
    return rate

def gradient(A, x, y):
    """ Gradient of the bilinear objective. """
    return A.dot(y), -np.transpose(A).dot(x)

def gradient_descent(A, x_init, y_init, figure, rate = 0.01, index = 0, iterations = 200):
    """ Gradient descent with a fixed stepsize. """
    x, y = x_init, y_init
    x_values, y_values = [], []

    for i in range(iterations):
        x_grad, y_grad = gradient(A, x, y)
        x = x - rate*(x_grad)
        x_values.append(x[index])
        y = y - rate*(y_grad)
        y_values.append(y[index])

    figure.plot(x_values, y_values, alpha=1,linewidth=0, label='gamma=0.5$', marker='x', color='b')
    figure.plot(x_values[0], y_values[0], alpha=1, linewidth=2, label='gamma=0.5$', marker='o', color='r')
    return x, y, i

def extragradient(A, d, x_init, y_init, rate, fig, figure, index = 0, max_iterations = 1500, adaptive = False, projection = projection):
    """ Extragradient descent. """
    x, y = x_init, y_init
    x_values, y_values = [], []
    duality_gap_err = []
    x_grad_ = 0
    i = 0

    while True:
        # Gradient step to go to an intermediate point.
        x_prev, y_prev = x, y
        x_grad, y_grad = gradient(A, x, y)
        # Calculate y_i.
        x_ = projection(x - rate*(x_grad),d)
        y_ = projection(y - rate*(y_grad),d)

        # Use the gradient of the intermediate point to perform a gradient step.
        x_grad_, y_grad_ = gradient(A, x_, y_)
        # Calculate x_i+1.
        x = projection(x - rate*(x_grad_),d)
        y = projection(y - rate*(y_grad_),d)

        x_values.append(x[index])
        y_values.append(y[index])
        if i % 4 == 0 or i >= max_iterations:
            # Check if the duality gap is 0 every 4 iterations.
            x_sample, y_sample = np.random.uniform(size=d), np.random.uniform(size=d)
            x_sample, y_sample = x_sample/np.sum(x_sample), y_sample/np.sum(y_sample)
            gap = ((x_sample.T @ A @ y) - (x.T @ A @ y_sample))**2
            duality_gap_err.append(gap)
            if (gap <= 0.00000001):
                break
        if adaptive == True:
            rate_prev = rate
            rate = linesearch_stepsize(A, x_prev, x_, x, x_grad_, y_prev, y_, y, y_grad_, rate_prev)
        i+=1

    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    ax2.plot(duality_gap_err, label='Duality gap')
    fig2.savefig("img/duality_gap")
    figure.plot(x_values, y_values, alpha=1,linewidth=0, label='gamma=0.5$', marker='x', color='g')
    figure.plot(x_values[0], y_values[0], alpha=1, linewidth=2, label='gamma=0.5$', marker='o', color='r')
    if adaptive == True: fig.savefig("img/adaptive_extragradient")
    else: fig.savefig("img/extragradient")
    return x, y, i