import cvxpy as cp
import numpy as np

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