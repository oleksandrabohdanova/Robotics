import numpy as np

from task.cost import Cost
from task.dynamics import Dynamics
from task.player import Player


def goal_cost(target_pos, x_size=6, u_size=2):
    """
    Cost function. Feel free to change it to achieve better scores!
    """
    # state: x, y, angle, vx, vy, vangle
    # Cost = x' M x - 2x' M tx + tx' M tx + u' u
    # (tx - x)'M(tx - x) = tx'Mtx - tx'Mx -x'Mtx +x'Mx =
    # = tx'Mtx - 2 x'Mtx + x'Mx
    # C_x = -2M(tx - x)
    # x'Mx - 2x'Mtx

    # C_x = 2 M (x - tx)
    # C_u = 2 u
    # C_xx = 2 M
    # C_uu = 2 I
    # C_xu = C_ux = 0

    M_diag = np.ones(x_size) * 100
    M_diag[x_size // 2 :] = 10.0
    M = np.diag(M_diag)

    target_state = np.zeros(x_size)
    target_state[: len(target_pos)] = target_pos

    C = lambda x, u: ((x - target_state).T @ M @ (x - target_state) + u.T @ u)
    C_x = lambda x, u: 2 * M @ (x - target_state)
    C_u = lambda x, u: 2 * u
    C_xx = lambda x, u: 2 * M
    C_uu = lambda x, u: 2 * np.eye(u_size)
    C_xu = lambda x, u: np.zeros((x_size, u_size))
    C_ux = lambda x, u: np.zeros((u_size, x_size))

    Cf = lambda x: (x - target_state).T @ M @ (x - target_state)
    Cf_x = lambda x: 2 * M @ (x - target_state)
    Cf_xx = lambda x: 2 * M
    return Cost(C, C_x, C_u, C_xx, C_ux, C_uu, Cf, Cf_x, Cf_xx)


class iLQR:
    def __init__(self, dynamics: Dynamics, cost: Cost):
        """
        iterative Linear Quadratic Regulator
        Args:
          dynamics: dynamics container
          cost: cost container
        """
        self.cost = cost
        self.dynamics = dynamics
        self.params = {
            "alphas": 0.5 ** np.arange(8),  # line search candidates
            "regu_init": 20,  # initial regularization factor
            "max_regu": 10000,
            "min_regu": 0.001,
        }

    def fit(self, x0, us_init, maxiters=50, early_stop=True):
        """
        Args:
          x0: initial state
          us_init: initial guess for control input trajectory
          maxiter: maximum number of iterations
          early_stop: stop early if improvement in cost is low.

        Returns:
          xs: optimal states
          us: optimal control inputs
          cost_trace: cost trace of the iterations
        """
        return run_ilqr(
            self.dynamics.f,
            self.dynamics.f_prime,
            self.cost.C,
            self.cost.Cf,
            self.cost.C_prime,
            self.cost.Cf_prime,
            x0,
            us_init,
            maxiters,
            early_stop,
            **self.params
        )

    def rollout(self, x0, us):
        """
        Args:
          x0: initial state
          us: control input trajectory

        Returns:
          xs: rolled out states
          cost: cost of trajectory
        """
        return rollout(self.dynamics.f, self.cost.C, self.cost.Cf, x0, us)


##########################################################
#                                                        #
#                ASSIGNMENT ENTRY POINT                  #
#                                                        #
##########################################################
def run_ilqr(
    f,
    f_prime,
    C,
    Cf,
    C_prime,
    Cf_prime,
    x0,
    u_init,
    max_iters,
    early_stop,
    alphas,
    regu_init=100,
    max_regu=10000,
    min_regu=0.00001,
):
    """
    iLQR main loop
    ----------------------------
    f - function f(x, u) -> x'
    f_prime - a tuple of Jf_x and Jf_u functions
    C - cost function c(x, u)
    Cf - cost function c(x) (final const)
    C_prime - cost function derivative (tuple of functions C_x, C_u, C_xx, C_ux, C_uu)
    Cf_prime - final cost function derivative (tuple of Cf_x, Cf_xx).
    x0 - initial state.
    u_init - initial plan.
    ----------------------------
    max_iters - maximum number of iterations
    early_stop - boolean, if True, stop if low error diff.
    alphas - list of alphas for linear search
    """
    us = u_init
    regu = regu_init

    #######################################
    # IMPLEMENT rollout                   #
    #######################################
    xs, J_old = rollout(f, C, Cf, x0, us)
    # cost trace
    cost_trace = [J_old]

    # Run main loop
    for it in range(max_iters):
        #######################################
        # IMPLEMENT backward_pass             #
        #######################################
        ks, Ks, exp_cost_redu = backward_pass(
            *f_prime, *C_prime, *Cf_prime, xs, us, regu
        )

        # Early termination if improvement is small
        if it > 3 and early_stop and np.abs(exp_cost_redu) < 1e-5:
            break

        # Backtracking line search
        for alpha in alphas:
            #######################################
            # IMPLEMENT forward_pass              #
            #######################################
            xs_new, us_new, J_new = forward_pass(f, C, Cf, xs, us, ks, Ks, alpha)
            if J_old - J_new > 0:
                # Accept new trajectories and lower regularization
                J_old = J_new
                xs = xs_new
                us = us_new
                regu *= 0.7
                break
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0

        cost_trace.append(J_old)
        regu = min(max(regu, min_regu), max_regu)

    return xs, us, cost_trace


##########################################################
#                                                        #
#                IMPLEMENT rollout function              #
#                                                        #
##########################################################
def rollout(f, C, Cf, x0, us):
    """
    Rollout with initial state and control trajectory

    Given cost function C and Cf, dynamics f, initial state x0 and plan us,
    compute all states xs and the final cost.
    """
    xs = np.empty((us.shape[0] + 1, x0.shape[0]))
    xs[0] = x0
    cost = 0
    ###############################################
    # Compute xs[1:] and cost
    ###############################################
    # YOUR CODE HERE
    for n in range(us.shape[0]):
        pass
    ###############################################
    return xs, cost


##########################################################
#                                                        #
#            IMPLEMENT forward_pass function             #
#                                                        #
##########################################################
def forward_pass(f, C, Cf, xs, us, ks, Ks, alpha):
    """
    Forward pass of iLQR given all the information and a full trajectory (plan + states).
    """
    xs_new = np.empty(xs.shape)

    cost_new = 0.0
    xs_new[0] = xs[0]
    us_new = us + alpha * ks

    ###############################################
    # Update us_new, compute xs_new and const_new here
    # Do not forget to add final cost
    ###############################################
    # YOUR CODE HERE
    for n in range(us.shape[0]):
        pass
    ###############################################

    return xs_new, us_new, cost_new


##########################################################
#                                                        #
#           IMPLEMENT backward_pass function             #
#                                                        #
##########################################################
#fmt: off
def backward_pass(
    f_x_prime, f_u_prime,
    C_x, C_u, C_xx, C_ux, C_uu,
    Cf_x, Cf_xx,
    xs, us,
    regu,
):
#fmt: on
    """
    Backward pass of iLQR given all the information and a full trajectory (plan + states).
    """
    ks = np.empty(us.shape)
    Ks = np.empty((us.shape[0], us.shape[1], xs.shape[1]))

    delta_V = 0
    V_x, V_xx = Cf_x(xs[-1]), Cf_xx(xs[-1])
    regu_I = regu * np.eye(V_xx.shape[0])
    for n in range(us.shape[0] - 1, -1, -1):

        x = xs[n]
        u = us[n]
        f_x, f_u = f_x_prime(x, u), f_u_prime(x, u)
        l_x, l_u, l_xx, l_ux, l_uu = (
            C_x(x, u),
            C_u(x, u),
            C_xx(x, u),
            C_ux(x, u),
            C_uu(x, u),
        )

        ###############################################
        # Compute Q_x, Q_u, Q_xx, Q_ux, Q_uu
        ###############################################
        # YOUR CODE HERE
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = None, None, None, None, None
        ###############################################

        # Compute gains with regularization. I provide the implementation
        # myself to not overcomplicate things, and yet be stable.
        f_u_dot_regu = f_u.T @ regu_I
        Q_ux_regu = Q_ux + f_u_dot_regu @ f_x
        Q_uu_regu = Q_uu + f_u_dot_regu @ f_u
        Q_uu_inv = np.linalg.inv(Q_uu_regu)

        k = -Q_uu_inv @ Q_u
        K = -Q_uu_inv @ Q_ux_regu
        ks[n], Ks[n] = k, K

        ###############################################
        # Compute V_x, V_xx
        ###############################################
        # YOUR CODE HERE
        V_x, V_xx = None, None
        ###############################################

        # expected cost reduction
        delta_V += Q_u.T @ k + 0.5 * k.T @ Q_uu @ k

    return ks, Ks, delta_V


class iLQRPlayer(Player):
    def __init__(self, dt, mass, gravity, Ixx, L, use_mpc=False):
        self.name = "iLQR" + ("_MPC" if use_mpc else "")
        self.alpha = 200
        self.anim_id = 3 if use_mpc else 2
        super().__init__()
        self.dynamics = Dynamics(
            dt=dt,
            mass=mass,
            gravity=gravity,
            Ixx=Ixx,
            L=L,
        )
        self.use_mpc = use_mpc
        self.T = 20
        self.L = L
        self.prev_target_counter = -1

    def act(self, target_pos, obs):
        target_pos = target_pos.astype(float)

        update_plan = False
        ###############################################
        # Implement self.use_mpc = True functionality.
        ###############################################
        # YOUR CODE HERE
        if self.use_mpc:
            pass
        ###############################################

        if self.target_counter != self.prev_target_counter:
            update_plan = True
            self.prev_target_counter = self.target_counter

        self.target_pos = target_pos

        if update_plan:
            self.ilqr = iLQR(dynamics=self.dynamics, cost=goal_cost(target_pos))
            self.i_plan = 0

        if self.i_plan == self.T or update_plan:
            us_init = np.zeros((self.T, 2))
            x0 = obs
            xs, self.plan, costs = self.ilqr.fit(
                x0, us_init, maxiters=10, early_stop=True
            )
            self.i_plan = 0

        F, M = self.plan[self.i_plan]
        self.i_plan += 1

        u1 = 0.5 * (F - M / self.L)
        u2 = 0.5 * (F + M / self.L)

        return u1, u2
