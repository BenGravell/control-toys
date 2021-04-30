from autograd import jacobian, hessian
import autograd.numpy as np
import autograd.numpy.linalg as la
from scipy.linalg import solve_continuous_are

from settings import period_eq, disturb


def care(A, B, Q, R, S=None):
    return solve_continuous_are(a=np.copy(A), b=np.copy(B), q=np.copy(Q), r=np.copy(R), s=np.copy(S))


def get_gain(A, B, Q, R, S):
    P = care(A, B, Q, R, S)
    K = -la.solve(R, B.T.dot(P) + S.T)
    return K


def linearize(dynamics, x, u):
    n, m = len(x), len(u)
    z = np.hstack([x, u])
    jac = jacobian(dynamics)
    J = jac(z)
    A, B = J[:, 0:n], J[:, n:n+m]
    return A, B


def quadratize(cost, x, u):
    n, m = len(x), len(u)
    z = np.hstack([x, u])
    hess = hessian(cost)
    H = hess(z)
    Q = H[0:n, 0:n]
    R = H[n:n+m, n:n+m]
    S = H[0:n, n:n+m]
    return Q, R, S


def set_eq(t, x_eq_list, u_eq_list, period_eq=period_eq):
    idx_eq = (int(t/period_eq)+1) % len(x_eq_list)
    return x_eq_list[idx_eq], u_eq_list[idx_eq]


def policy(x, x_eq, u_eq, K):
    dx = x - x_eq
    du = np.dot(K, dx)
    u = u_eq + du
    return u


def step(x, policy, policy_params, dynamics, disturbance, stepsize, method='euler'):
    u = policy(x, **policy_params)
    z = np.hstack([x, u])
    w = disturbance(z) if disturb else np.zeros_like(x)
    if method == 'euler':
        x = x + stepsize*dynamics(z)
    elif method == 'rk4':
        x1 = np.copy(x)
        u1 = np.copy(u)
        z1 = np.hstack([x1, u1])
        k1 = dynamics(z1)

        x2 = x1 + stepsize*k1/2
        u2 = policy(x2, **policy_params)
        z2 = np.hstack([x2, u2])
        k2 = dynamics(z2)

        x3 = x2 + stepsize*k2/2
        u3 = policy(x3, **policy_params)
        z3 = np.hstack([x3, u3])
        k3 = dynamics(z3)

        x4 = x3 + stepsize*k3
        u4 = policy(x4, **policy_params)
        z4 = np.hstack([x4, u4])
        k4 = dynamics(z4)

        x = x + (stepsize/6)*(k1 + 2*k2 + 2*k3 + k4)
    x = x + (stepsize**0.5)*w
    return x, u
