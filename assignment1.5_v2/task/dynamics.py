import numba
import numpy as np


@numba.njit(cache=True, fastmath=True)
def _finite_diff(fun, x, u, i, eps, params):
    """
    Finite difference approximation
    """

    args = (x, u)
    fun0 = fun(x, u, *params)

    m = x.size
    n = args[i].size

    Jac = np.zeros((m, n))
    for k in range(n):
        args[i][k] += eps
        Jac[:, k] = (fun(args[0], args[1], *params) - fun0) / eps
        args[i][k] -= eps

    return Jac


def jacobian(f, params, eps=1e-4):
    """
    Construct from a discrete time dynamics function
    """
    f_x = lambda x, u: _finite_diff(f, x, u, 0, eps, params)
    f_u = lambda x, u: _finite_diff(f, x, u, 1, eps, params)
    return f_x, f_u


@numba.njit
def _x_dot(state, action, mass, Ixx, gravity):
    F, M = action
    _, _, angle, vx, vy, angle_vel = state
    F_clamped, M_clamped = F, M

    # First derivative, xdot = [vy, vz, phidot, ay, az, phidotdot]
    xdot = np.array(
        [
            vx,
            vy,
            angle_vel,
            -F_clamped * np.sin(np.deg2rad(angle)) / mass,
            -F_clamped * np.cos(np.deg2rad(angle)) / mass + gravity,
            M_clamped / Ixx,
        ]
    )
    return xdot


class Dynamics:
    def __init__(self, dt, mass, gravity, Ixx, L):
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        self.Ixx = Ixx
        self.L = L

        @numba.njit
        def _f(state, action, dt, mass, Ixx, gravity):
            return state + _x_dot(state, action, mass, Ixx, gravity) * dt

        self.J_f_state, self.J_f_action = jacobian(_f, params=(dt, mass, Ixx, gravity))

    # Limit force and moment to prevent saturating the motor
    # Clamp F and M such that u1 and u2 are between 0 and 1.7658
    #
    #    u1      u2
    #  _____    _____
    #    |________|
    #
    # F = u1 + u2
    # M = (u2 - u1)*L
    def _clamp(self, F, M):
        u1 = 0.5 * (F - M / self.L)
        u2 = 0.5 * (F + M / self.L)

        if u1 < 0 or u1 > 1.7658 or u2 < 0 or u2 > 1.7658:
            print(f"motor saturation {u1} {u2}")

        u1_clamped = min(max(0, u1), 1.7658)
        u2_clamped = min(max(0, u2), 1.7658)
        u1_clamped = u1
        u2_clamped = u2
        F_clamped = u1_clamped + u2_clamped
        M_clamped = (u2_clamped - u1_clamped) * self.L

        return F_clamped, M_clamped

    # Equation of motion
    # dx/dt = f(t, x)
    #
    # t     : Current time (seconds), scalar
    # x     : Current state, [y, z, phi, vy, vz, phidot]
    # return: First derivative of state, [vy, vz, phidot, ay, az, phidotdot]

    def x_dot(self, state, action):
        return _x_dot(state, action, self.mass, self.Ixx, self.gravity)

    def f(self, state, action):
        return state + self.x_dot(state, action) * self.dt

    @property
    def f_prime(self):
        return self.J_f_state, self.J_f_action
