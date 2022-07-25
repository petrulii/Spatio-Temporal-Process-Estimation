import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from projections import projection_cvx, projection_simplex_sort, projection_Moreau

def theoretical_stepsize(H):
    """ Calculates the theoretical optimal step size based on the Hessian of size n*m of the bilinear objective function. """
    U, s, VT = linalg.svd(H)
    return 1/np.sqrt(np.max(s))

def linesearch_stepsize(A, x_i, y_i, x_i_1, grad_y_i, x2_i, y2_i, x2_i_1, grad_y2_i, rate):
    """ Backtrack line search for step size. """
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

def matrix_game_gradient(A, x, y):
    """ Gradient of a min max bilinear objective for players X and Y. """
    return A.dot(y), -np.transpose(A).dot(x)

def gradient_descent(A, x_init, y_init, figure = None, rate = 0.01, index = 0, iterations = 1000, gradient = matrix_game_gradient):
    """ Gradient descent with a fixed stepsize.
    param A: the pay-off matrix A of dimensions d*d describing the bilinear min max objective
    param x_init: initial strategy vector for player X
    param y_init: initial strategy vector for player Y
    """
    x, y = x_init, y_init
    x_values, y_values = [], []

    for i in range(iterations):
        x_grad, y_grad = gradient(A, x, y)
        x = x - rate*(x_grad)
        x_values.append(x[index])
        y = y - rate*(y_grad)
        y_values.append(y[index])

    if figure != None:
        # Plot all descent points.
        figure.plot(x_values, y_values, alpha=1,linewidth=0, marker='x', color='b', label="Gradient descent")
        # Display the last point.
        figure.plot(x_values[-1], y_values[-1], alpha=1, linewidth=2, marker='o', color='r')
    return x, y, i

def extragradient(A, d, x_init, y_init, rate, figure = None, index = 0, max_iterations = 200, adaptive = False, projection = projection_cvx, gradient = matrix_game_gradient):
    """ Extragradient descent.
    param A: the pay-off matrix A of dimensions d*d describing the bilinear min max objective
    param x_init: initial strategy vector for player X
    param y_init: initial strategy vector for player Y
    """
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
        if i >= max_iterations:
            # Add i % 4 == 0 if want to check the duality gap every 4 iterations.
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
    if figure != None:
        if adaptive == True and projection==projection_simplex_sort:
            mrk='x'
            clr='g'
            lbl="EG + linesearch step + sort"
        elif adaptive == True and projection==projection_Moreau:
            mrk='*'
            clr='y'
            lbl="EG + linesearch step + projection_Moreau"
        else:
            mrk='x'
            clr='g'
            lbl="EG + theor. step"
        # Plot all descent points.
        figure.plot(x_values, y_values, alpha=1,linewidth=0, marker=mrk, color=clr, label=lbl)
        # Plot the last point.
        figure.plot(x_values[-1], y_values[-1], alpha=1, linewidth=2, marker='o', color='r')
        fig2 = plt.figure(2)
        ax2 = fig2.gca()
        ax2.plot(duality_gap_err, label='Duality gap')
        fig2.savefig("img/duality_gap")
    return x, y, i