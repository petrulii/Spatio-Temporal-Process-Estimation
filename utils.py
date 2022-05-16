import cvxpy as cp
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig



def l2_norm(z):
    """ Calculates the L2 norm of a given vector z. """
    return 1/2 * cp.norm2(z)**2


def projection_cvx(y, d):
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
    # If cond is False everywhere = no feasible bases.
    if cond.astype(int).sum() == 0:
        return np.zeros(d)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(y - theta, 0)
    return w


def projection_Moreau(y, n):
    """ Calculates the projection from point y onto a point x in the probability simplex.
    See Yunmei Chen and Xiaojing Ye, "Projection Onto a Simplex", 
    https://arxiv.org/abs/1101.6081 """
    y_s = sorted(y, reverse=True)
    sum_y = 0
    for i, y_i, y_next in zip(range(1, n+1), y_s, y_s[1:]+[0.0]):
        sum_y += y_i
        t = (sum_y-1)/i
        if t>=y_next:
            break
    return np.array([max(0, y_i-t) for y_i in y])


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


def matrix_game_gradient(A, x, y):
    """ Gradient of the bilinear objective. """
    return A.dot(y), -np.transpose(A).dot(x)


def gradient_descent(A, x_init, y_init, figure = None, rate = 0.01, index = 0, iterations = 1000, gradient = matrix_game_gradient):
    """ Gradient descent with a fixed stepsize. """
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
        if i >= max_iterations:#if i % 4 == 0 or i >= max_iterations:
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
        # Display the last point.
        figure.plot(x_values[-1], y_values[-1], alpha=1, linewidth=2, marker='o', color='r')
        fig2 = plt.figure(2)
        ax2 = fig2.gca()
        ax2.plot(duality_gap_err, label='Duality gap')
        fig2.savefig("img/duality_gap")
    return x, y, i

# Activation function.
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Loss function.
def log_loss(y, pred):
    return (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()

# Logistic loss gradient.
def log_loss_gradient(X, y, y_pred, n):
    """ Gradient of the negative logistic loss. """
    return 1/n * np.dot(X, (y_pred - y))

def gradient_descent(X, y, n, theta_init, rate, max_error = 0.5, max_iterations = 200, adaptive = False):
    """ Gradient descent. """
    theta = theta_init
    y_pred = np.zeros(n)
    log_err = [log_loss(y, y_pred)]
    i = 0

    while i < max_iterations or log_err[i] <= max_error:
        # Gradient step to go to an intermediate point.
        for j in range(n):
            y_pred[j] = 1/(1+np.exp(-np.dot(X.T, theta[j])))
            theta_grad = log_loss_gradient(X, y, y_pred[j], n)
            theta[j] = theta[j] - rate * theta_grad
        log_err.append(log_loss(y, y_pred))
        if adaptive == True:
            # Do this.
            print("Soryy, adatptive step size not here yet :(")
            return
        i+=1

    return theta, i, log_err

# Projected gradient.
def projected_gradient(X, y, n, theta_init, rate, figure = None, max_error = 0.5, max_iterations = 200, adaptive = False, projection = projection_cvx, gradient = log_loss_gradient):
    """ Projected gradient descent. """
    theta = theta_init
    y_pred = np.zeros(n)
    log_err = [log_loss(y, y_pred)]
    i = 0

    while i < max_iterations or log_err[i] <= max_error:
        y_pred = sigmoid(np.dot(X.T, theta))
        log_err.append(log_loss(y, y_pred))
        # Gradient step to go to an intermediate point.
        theta_grad = gradient(X, y, y_pred, n)
        # Calculate x_i+1 by projecting back onto the set
        theta = projection(theta - rate*theta_grad, n)
        if adaptive == True:
            # Do this.
            print("Soryy, adatptive step size not here yet :(")
            return
        i+=1

    return theta, i, log_err