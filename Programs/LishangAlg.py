import numpy as np
# from matplotlib import pyplot as plt
from scipy.stats import norm
from numba import njit
import pandas as pd


# we can simulate the same datasets as Lishang et al.
x = 10
R = 0.12                        # R = r
D = 0.                          # D = q in Lishang, assume both are known
tau = 1
L, M = 1.8,1.8
a_bounds = (0.005, 0.045)

tau_steps = 504                 # 252 * 2
x_steps = 1000

x_domain = np.linspace(0, L+M, x_steps)
y_domain = x_domain - L
K_path = x * np.exp(y_domain)

s_domain = x * np.exp(y_domain)
sigma = ((np.log(s_domain) - np.log(10))**2) / 40 + 0.2
a_true = sigma**2 / 2
a_low, a_high = a_bounds

sigma = 0.25
v = np.zeros(x_steps)           # this becomes our obtained K curve
x_path = K_path / np.log(y_domain)
x_path
for i in range (x_steps):
        
    d1 = (np.log(x/K_path[i]) + (R - D + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
    d2 = (np.log(x/K_path[i]) + (R - D - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
    
    price = (x * np.exp(-D * tau) * norm.cdf(d1) - K_path[i] * np.exp(-R * tau) * norm.cdf(d2))
    
    v = (np.exp(D * tau) / x) * price

v    

def v_solver_implicit (R, D, tau, a, bounds = (10,10), x_steps = 1000, tau_steps = 504):
    """Implementation of the (Lishang et al., 2003) Meyer solving algorithm for equation (3.1).

    args: R - risk-free rate
          D - dividend rate
          tau - time till maturity for the option
          a - the volatility vector
          bounds - duple of (L,M)
          x_steps - number of steps in space
          tau_steps - the number of steps in tau

    returns: v - fitted valuation function"""

    # space variables
    L, M = bounds

    x = np.linspace (0., L+M, x_steps)
    dx = x[1] - x[0]
    
    # time variable
    delta_tau = tau / tau_steps

    # defining v
    v_last = np.maximum (1. - np.exp(x-L), 0.) # (A.1) v_0 (x) = max{1-e^y}
    v_past = np.empty((tau_steps+1, x_steps))  # holds v(x, 0 -> \tau^*)
    v_past[0] = v_last

    # Meyer's method: v = Uz + u
    U = np.zeros(x_steps)
    z = np.zeros(x_steps)
    u = np.zeros(x_steps)

    for t in range (tau_steps):
        """Loops x solving algorithm until we reach tau^*"""

        # initial values
        U[0] = 0                # (A.3)
        u[0] = 1                # (A.4)

        for i in range (1, x_steps):
            """Initial loop runs forwards from X(0 -> L+M), fills up U and u."""

            alpha = 1 / (a[i] * delta_tau) # required constants
            beta = (a[i] + R - D) / a[i]

            # implicit Euler scheme
            A, B, C = alpha * dx, 1 + beta * dx, - U[i-1] - dx
            U[i] = (- B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

            u[i] = (u[i-1] + dx * alpha * U[i] * v_last[i]) / (1 + dx * alpha * U[i])


        # initial condition for z
        z[-1] = - u[-1] / U[-1] # (A.5)

        for i in range (x_steps - 2, -1, -1):
            """Backwards loop running from X(L+M -> 0), fills up z."""

            lambda_ = (U[i] / (a[i] * delta_tau)) + ((a[i] + R - D) / a[i])
            gamma = (u[i] - v_last[i]) / (a[i] * delta_tau)

            z[i] = (z[i+1] - gamma * dx) / (1 + dx * lambda_)

        v_next = U * z + u
        v_past[t+1] = v_next
        v_last = v_next.copy()

    return v_past

def phi_solver_implicit (v_true, v_mod, a, R, D, tau, x, bounds = (10, 10), x_steps = 1000, tau_steps = 504):
    """Implementation of the (Lishang et al., 2003) Meyer solving algorithm for equation (3.1).

    args: v_true - market distributions of premiums across strike values
          v_mod - v(y, tau^*)
          a - volatility vector
          R - risk-free rate
          D - dividend rate
          tau - time till maturity
          x - underlying stock price
          bounds - duple of (L,M)
          x_steps - the number of steps in space
          tau_steps - the number of steps in tau

    returns: phi - goodness of fit variable"""

    # space variables
    L, M = bounds

    # x = np.linspace (0., L+M, phi_steps)
    # dx = x[1] - x[0]

    # time variable
    delta_tau = tau / tau_steps

    # Global variables
    mu = R - D
    a_y = np.gradient(a, x, edge_order=2)

    # phi list
    phi_last = v_mod - v_true
    phi_last[0], phi_last[-1] = 0, 0

    past_phi = np.empty((tau_steps+1, x_steps)) # of use in next step
    past_phi[0] = phi_last

    # Meyer's method: v = Uz + u
    U = np.zeros(x_steps)
    z = np.zeros(x_steps)
    u = np.zeros(x_steps)

    for t in range (tau_steps):
        """Loops x solving algorithm until we reach tau^*"""

        # initial values
        U[0] = 0                # (A.3)
        u[0] = 0                # (A.4)

        for i in range (1, x_steps):
            """Initial loop runs forwards from X(0 -> L+M), fills up U and u."""

            dx = x[i] - x[i-1]
            # implicit Euler scheme
            alpha = ((mu * a_y[i]) / a[i]) + 1 / delta_tau
            beta = 1 + (mu / a[i]) - (a_y[i] / a[i])

            A, B, C = dx * alpha, 1 - dx * beta, -U[i-1] - dx / a[i]
            U[i] = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

            c = U[i] * alpha + (a_y[i] / a[i])
            d = (phi_last[i] / delta_tau) * U[i]

            u[i] = (u[i-1] + dx * d) / (1 + dx * c)


        # initial condition for z
        z[-1] = - u[-1] / U[-1] # (A.5)

        for i in range (x_steps - 2, -1, -1):
            """Backwards loop running from X(L+M -> 0), fills up z."""

            dx = x[i+1] - x[i]

            lambda_ = (U[i] / delta_tau) - 1 - mu * ((1 - a_y[i] * U[i]) / a[i])
            gamma = ((u[i] - phi_last[i]) / delta_tau) + ((a_y[i] * u[i]) / a[i]) * mu

            z[i] = (z[i+1] - dx * gamma) / (1 + dx * lambda_) 



        phi_next = U * z + u
        past_phi[t+1] = phi_next
        phi_last = phi_next.copy()

    return past_phi[::-1]


def variationalineqsolver (past_phi, past_v, R, D, tau, x_domain, a, a_0, a_1, N_reg, tau_steps,
                           alpha = 0.45, eta = 1e-4, max_steps = 2000):
    """Implementation of the variational inequality scheme in (Lishang et al. 2003)

    args:
    past_phi - all values of phi(y,0 -> tau^*)
    past_v - all values of v(y,0 -> tau^*)
    R - risk-free rate
    D - dividend rate
    tau - time till maturity
    x_domain - the map of space (doesn't matter between x and y, same map different origin)
    a - transformed volatility estimate
    a_0 - minimum transformed volatility estimate
    a_1 - maximum transformed volatility estimate
    N_dat -
    tau_steps - number of steps in tau
    alpha - estimate for alpha
    
    """

    dy = x_domain[1] - x_domain[0]
    dtau = tau / tau_steps

    # constructing f(y; v, \phi)
    # first derivatives of v(y,tau^*)
    v_tau = np.gradient(past_v, dtau, axis=0, edge_order=2)
    v_y = np.gradient(past_v, x_domain, axis=1, edge_order=2)


    # integration step
    integrand = (past_phi * (v_tau + (R-D) * v_y) / a[None,:])
    f = np.trapezoid(integrand, dx=dtau, axis=0) / N_reg


    # (A.7)
    dt = alpha * dy**2

    a_old = a.copy()

    for t in range(max_steps):
        
        B = alpha * a_old[2:] + (1 - 2 * alpha) * a_old[1:-1] + alpha * a_old[:-2] - f[1:-1] * dt

        a_new = a_old.copy()
        
        a_new[1:-1] = np.clip(B, a_0, a_1)

        a_new[0], a_new[-1] = a_new[1], a_new[-2]

        a_old = a_new

    return a_old

def LishangInversion(
    v_true,
    a,
    R,
    D,
    tau,
    y_true,
    space_bounds,
    a_bounds,
    stock_price,
    N_regl=1e-2,
    vieta=1e-4,
    outer_eta=1e-7,
    max_steps_alpha=5000,
    x_steps=1000,
    tau_steps=504,
    alpha = 0.45,
    max_outer_loops=600,
):
    """Wrapper inversion function"""

    if isinstance(a, (int, float)):
        a = np.ones(x_steps) * a
    elif np.isscalar(a):
        a = np.ones(x_steps) * np.float64(a)
    else:
        a = np.asarray(a, dtype=np.float64)

    L, M = space_bounds
    a_low, a_high = a_bounds

    x_domain = np.linspace(0, L + M, x_steps)
    y_domain = x_domain - L

    s_idx = np.argsort(y_true, kind="stable")

    y_sorted, v_sorted = y_true[s_idx], v_true[s_idx]

    y_bound = np.concatenate(([-L], y_sorted, [M]))
    v_bound = np.concatenate(([1], v_sorted, [0]))

    # interpolating empirical data across discretised values
    v_true_grid = np.interp(y_domain, y_bound, v_bound)


    for step in range (max_outer_loops):
    # modelled premiums
        v_past = v_solver_implicit(R, D, tau, a, space_bounds, x_steps, tau_steps)
        v_tau_star = v_past[-1]
    
        phi_past = phi_solver_implicit(v_true_grid, v_tau_star, a, R, D, tau, x_domain,
                                       space_bounds, x_steps, tau_steps)
    
        new_a = variationalineqsolver(phi_past, v_past, R, D, tau, x_domain, a,
                                      a_low, a_high, N_regl, tau_steps, alpha, vieta, max_steps_alpha)

        if np.linalg.norm(a - new_a, ord=np.inf) < outer_eta:
            return new_a

        a = new_a.copy()

    return new_a

y = np.log(K_path / x)

# v is the model set of insurance premiums, 0.02 is the first volatility estimate, R - D is \mu,
# tau is the time till maturity, y = ln(K/x),
first_sigma_est = 0.25
first_a_est = (first_sigma_est**2) / 2

est_a = LishangInversion (v, first_a_est, R, D, tau, y, (L, M), (0.005, 0.045), x)

# est_a = 0.045

est_sigma = np.sqrt(2 * est_a)
est_sigma
sigma
np.mean(est_sigma)

# testing smooth =====================================
L, M = 10, 10

x_domain = np.linspace(
    0.0,
    L + M,
    x_steps
)

y_domain = x_domain - L

sigma_true = 0.25

a_true = np.full(
    x_steps,
    0.5 * sigma_true**2
)

v_true = v_solver_implicit(
    R,
    D,
    tau,
    a_true,
    (L, M),
    x_steps,
    tau_steps
)[-1]


est_a = LishangInversion(
    v_true=v_true,
    a=0.5 * 0.20**2,
    R=R,
    D=D,
    tau=tau,
    y_true=y_domain,
    space_bounds=(L, M),
    a_bounds=(0.005, 0.045),
    stock_price=10.0,
    N_regl=1.0,
    outer_eta=1e-7,
    max_outer_loops=600,
    x_steps=x_steps,
    tau_steps=tau_steps,
)

est_sigma = np.sqrt(2 * est_a)

np.mean(est_sigma)

