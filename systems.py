from functools import partial
import autograd.numpy as np
import autograd.numpy.random as npr
from control import linearize, quadratize, get_gain, policy, step, set_eq
from plotting import make_style_kwargs, setup_plot, draw_system_parametric

from settings import step_size, step_method


class ToySystem:
    def __init__(self, system_str, draw=True):
        self.system_str = system_str
        for key, value in make_system_data(system_str).items():
            setattr(self, key, value)

        self.x, self.u = self.x_eq_list[0], self.u_eq_list[0]
        A, B = linearize(self.dynamics, self.x, self.u)
        Q, R, S = quadratize(self.cost, self.x, self.u)
        self.lqr_data = dict(A=A, B=B, Q=Q, R=R, S=S)
        self.K = get_gain(**self.lqr_data)

        if draw:
            self.init_plot()

    def init_plot(self):
        self.fig, self.ax = setup_plot(self.system_str)
        self.draw_system = partial(draw_system_parametric, make_artist_props=self.make_artist_props)
        self.artists = self.draw_system(self.x, self.u, self.x, self.u, self.ax, artists=None)

    def update(self, t):
        x_eq, u_eq = set_eq(t, self.x_eq_list, self.u_eq_list)
        policy_params = dict(x_eq=x_eq, u_eq=u_eq, K=self.K)
        self.x, self.u = step(self.x, policy, policy_params, self.dynamics, self.disturbance,
                              step_size=step_size, method=step_method)
        return self.draw_system(self.x, self.u, x_eq, u_eq, self.ax, self.artists)


def make_system_data(system_str=None):
    if system_str == 'cartpole':
        # States
        # x[0]  Cart position
        # x[1]  Pole angle
        # x[2]  Cart velocity
        # x[3]  Pole angular velocity

        # Inputs
        # u[0]  Force on cart

        n, m = 4, 1

        gravity = 9.81
        length = 1.2
        moment_of_inertia = 1.00
        mass_pole = 1.00
        mass_cart = 1.00
        damping = 5.00

        width_cart = 0.40

        def dynamics(z):
            x, u = z[0:n], z[n:n+m]
            s, c = np.sin(x[1]), np.cos(x[1])

            c0 = (mass_pole * length) / (moment_of_inertia+mass_pole * length * length)
            c1 = c0 * mass_pole * length
            c2 = 1 / (mass_cart+mass_pole-c1 * c * c)
            c3 = c1 * gravity * s * c-mass_pole * length * x[3] * x[3] * s-damping * x[2]

            f3 = c2 * c3

            return np.array([x[2], x[3], f3+c2 * u[0], c0 * (gravity * s+f3 * c)])

        def cost(z):
            Q = np.diag([100, 1, 10, 1])
            R = np.diag([10])
            H = np.block([[Q, np.zeros([n, m])], [np.zeros([m, n]), R]])
            return np.dot(z, np.dot(H, z))

        def disturbance(z):
            mean = np.zeros(n)
            covr = np.diag([0, 0, 0.001, 0.001])
            return npr.multivariate_normal(mean, covr)

        def make_artist_props(x, u):
            p1 = np.array([x[0], 0])
            p2 = p1 + length * np.array([np.sin(x[1]), np.cos(x[1])])

            q1 = p1 + np.array([width_cart/2, 0])
            q2 = p1 + np.array([-width_cart/2, 0])

            d1 = make_style_kwargs(q1, q2, lw=12, style='base')
            d2 = make_style_kwargs(p1, p2, lw=8, style='accent1')
            return d1, d2

        x_eq_list = [np.array([0, 0, 0, 0]),
                     np.array([0.88, 0, 0, 0]),
                     np.array([-0.72, 0, 0, 0]),
                     np.array([0.46, 0, 0, 0]),
                     np.array([-0.27, 0, 0, 0])]
        u_eq_list = [np.zeros(m) for _ in x_eq_list]

    elif system_str == 'ballbeam':
        # States
        # x[0]  Ball position
        # x[1]  Beam angle
        # x[2]  Ball velocity
        # x[3]  Beam angular velocity

        # Inputs
        # u[0]  Motor torque

        n, m = 4, 1

        moment_of_inertia = 1.0
        inertia_inv = 1 / moment_of_inertia
        gravity = 9.81
        damping = 0.1

        beam_radius = 1.2
        ball_height = 0.2

        def dynamics(z):
            x, u = z[0:n], z[n:n+m]
            s = np.sin(x[1])

            return np.array([x[2],
                             x[3],
                             -gravity*s - damping*x[2],
                             inertia_inv*u[0]])

        def cost(z):
            Q = np.diag([1, 1, 1, 1])
            R = np.diag([4])
            H = np.block([[Q, np.zeros([n, m])], [np.zeros([m, n]), R]])
            return np.dot(z, np.dot(H, z))

        def disturbance(z):
            mean = np.zeros(n)
            covr = 0.01*np.diag([0, 0, 0.01, 0.0002])
            return npr.multivariate_normal(mean, covr)

        def make_artist_props(x, u):
            s, c = np.sin(-x[3]), np.cos(x[3])
            e = np.array([c, s])
            d = np.array([-s, c])
            p0 = np.array([0, 0])

            p1 = p0 - beam_radius * e
            p2 = p0 + beam_radius * e

            q1 = p0 + x[0] * e
            q2 = q1 + ball_height * d

            d1 = make_style_kwargs(p1, p2, lw=12, style='base')
            d2 = make_style_kwargs(q1, q2, lw=10, style='accent1')
            return d1, d2

        x_eq_list = [np.array([0, 0, 0, 0]),
                     np.array([0.88, 0, 0, 0]),
                     np.array([-0.32, 0, 0, 0]),
                     np.array([0.46, 0, 0, 0]),
                     np.array([-0.77, 0, 0, 0])]
        u_eq_list = [np.zeros(m) for _ in x_eq_list]

    elif system_str == 'quadrotor':
        # States
        # x[0]  Horizontal position
        # x[1]  Vertical position
        # x[2]  Angle
        # x[3]  Horizontal velocity
        # x[4]  Vertical velocity
        # x[5]  Angular velocity

        # Inputs
        # u[0]  Summed thrust, acts upward
        # u[1]  Differential thrust, acts counterclockwise

        n, m = 6, 2

        mass = 1.0
        mass_inv = 1 / mass
        radius = 0.4
        moment_of_inertia = 1.0
        inertia_inv = radius / moment_of_inertia
        gravity = 9.81

        rotor_height = 0.5*radius
        rotor_radius = 0.7*radius

        def dynamics(z):
            x, u = z[0:n], z[n:n+m]
            s, c = np.sin(x[2]), np.cos(x[2])

            return np.array([x[3],
                             x[4],
                             x[5],
                             -mass_inv * u[0] * s,
                             mass_inv * u[0] * c - gravity,
                             inertia_inv * u[1]])

        def cost(z):
            Q = np.diag([10, 10, 1, 1, 1, 1])
            R = np.diag([4, 4])
            H = np.block([[Q, np.zeros([n, m])], [np.zeros([m, n]), R]])
            return np.dot(z, np.dot(H, z))

        def disturbance(z):
            mean = np.zeros(n)
            covr = np.diag([0.0002, 0.0001, 0.0001, 0.004, 0.002, 0.002])
            return npr.multivariate_normal(mean, covr)

        def make_artist_props(x, u):
            s, c = np.sin(x[2]), np.cos(x[2])
            e = np.array([c, s])
            d = np.array([-s, c])
            p0 = x[0:2]

            p1 = p0 - radius * e
            p2 = p0 + radius * e

            q11 = p0 - 0.95*radius * e
            q12 = q11 + rotor_height * d
            q21 = p0 + 0.95*radius * e
            q22 = q21 + rotor_height * d

            r11 = q12 - rotor_radius*e
            r12 = q12 + rotor_radius*e
            r21 = q22 - rotor_radius*e
            r22 = q22 + rotor_radius*e

            d1 = make_style_kwargs(p1, p2, lw=12, style='base')
            d2 = make_style_kwargs(q11, q12, lw=8, style='accent1')
            d3 = make_style_kwargs(q21, q22, lw=8, style='accent2')
            d4 = make_style_kwargs(r11, r12, lw=4, style='accent1')
            d5 = make_style_kwargs(r21, r22, lw=4, style='accent2')

            return d1, d2, d3, d4, d5

        x_eq_list = [np.array([0, 0, 0, 0, 0, 0]),
                     np.array([0.39, 0.27, 0, 0, 0, 0]),
                     np.array([-0.47, 0.67, 0, 0, 0, 0]),
                     np.array([0.57, -0.90, 0, 0, 0, 0]),
                     np.array([-0.46, -0.79, 0, 0, 0, 0])]
        u_eq_list = [np.array([mass*gravity, 0]) for _ in x_eq_list]

    elif system_str == 'bicycle':
        # States
        # x[0]  Horizontal position deviation
        # x[1]  Vertical position deviation
        # x[2]  Body angle

        # Inputs
        # u[0]  Forward velocity deviation
        # u[1]  Steering angle

        n, m = 3, 2

        body_length = 1.0
        nominal_velocity = 1.0

        wheel_length = 0.2*body_length

        def dynamics(z):
            x, u = z[0:n], z[n:n+m]
            wheel_angle = x[2] + u[1]
            s, c = np.sin(wheel_angle), np.cos(wheel_angle)
            forward_velocity = nominal_velocity + u[0]

            return np.array([-forward_velocity*s,
                             forward_velocity*c - nominal_velocity,
                             (forward_velocity/body_length)*np.sin(u[1])])

        def cost(z):
            Q = np.diag([1, 1, 1])
            R = np.diag([2, 5])
            H = np.block([[Q, np.zeros([n, m])], [np.zeros([m, n]), R]])
            return np.dot(z, np.dot(H, z))

        def disturbance(z):
            mean = np.zeros(n)
            covr = np.diag([0.0002, 0.001, 0.0001])
            return npr.multivariate_normal(mean, covr)

        def make_artist_props(x, u):
            body_angle = x[2]
            wheel_angle = x[2] + u[1]
            sb, cb = np.sin(body_angle), np.cos(body_angle)
            sw, cw = np.sin(wheel_angle), np.cos(wheel_angle)

            e = np.array([-sb, cb])
            d = np.array([-sw, cw])
            p0 = x[0:2]

            p1 = p0
            p2 = p0 - body_length * e

            q11 = p1 - wheel_length * d
            q12 = p1 + wheel_length * d
            q21 = p2 - wheel_length * e
            q22 = p2 + wheel_length * e

            d1 = make_style_kwargs(p1, p2, lw=16, style='base')
            d2 = make_style_kwargs(q11, q12, lw=8, style='accent1')
            d3 = make_style_kwargs(q21, q22, lw=8, style='accent2')
            return d1, d2, d3

        x_eq_list = [np.array([0, 0, 0]),
                     np.array([0.59, 0.87, 0]),
                     np.array([-0.37, 0.67, 0]),
                     np.array([0.97, 0.10, 0]),
                     np.array([-0.46, 0.39, 0])]
        u_eq_list = [np.zeros(m) for _ in x_eq_list]

    else:
        raise ValueError

    return dict(dynamics=dynamics,
                cost=cost,
                disturbance=disturbance,
                x_eq_list=x_eq_list,
                u_eq_list=u_eq_list,
                make_artist_props=make_artist_props)
