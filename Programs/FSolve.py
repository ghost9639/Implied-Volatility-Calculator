from scipy.optimize import least_squares
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# from random import randint
# randint(0, steps)



# variable selection from Bouchouev
x = 710.82
tau = 43/252
R = 0.05
D = 0.02


# variable selection from me
sigma = 0.25
K = 730
steps = 5000

path = np.linspace(x-300, x+300, steps)
prices = np.zeros(steps, dtype=np.float64)

d1 = (np.log(path/K) + (R - D + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
d2 = (np.log(path/K) + (R - D - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))

prices = (path * np.exp(-D * tau) * norm.cdf(d1) - K * np.exp(-R * tau) * norm.cdf(d2))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.plot(path, prices)
plt.title("Value of option premium across initial stock price")
plt.xlabel(r"Stock Price $x$")
plt.ylabel(r"Option Premium $u(x,0)$")

plt.show()

# for our case, we only want to consider a system of three equations for our inverted parameters

# calculated premiums

def d1_calc (x, K, tau, sigma, R, D):
    """Calculation utility for d1"""
    return ((np.log(x/K) + (R - D + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau)))

def d2_calc (x, K, tau, sigma, R, D):
    """Calculation utility for d2"""
    return ((np.log(x/K) + (R - D - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau)))

def premium_calc (x_v, tau_v, K_v, sigma_v, R_v, D_v):
    """Calculation utility for the call option premium"""

    d1_v = (np.log(x_v / K_v) + (R_v - D_v + 0.5 * (sigma_v**2)) * tau_v) / (sigma_v * np.sqrt(tau_v))
    d2_v = (np.log(x_v / K_v) + (R_v - D_v - 0.5 * (sigma_v**2)) * tau_v) / (sigma_v * np.sqrt(tau_v))

    return (x_v * np.exp(-D_v * tau_v) * norm.cdf(d1_v) - K_v * np.exp(-R_v * tau_v) * norm.cdf(d2_v)) 



# varying x ===========================================

test_x = np.array([670, 700, 730])
tau_v, K_v = 43/252, 730
test_x_miss_prem = premium_calc(test_x, tau_v, K_v, sigma, R, D)


# assume we get an initial estimate and return [sigma, R, D]
def price_calc_x(arr):
    """Provides series of 3 equations when x varies, extremely scuffed, assumes you've already set test_x_miss_prem"""

    sigma_v, R_v, D_v = arr


    p1 = (test_x[0] * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(test_x[0], K_v, tau_v, sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(test_x[0], K_v, tau_v, sigma_v, R_v, D_v))) - test_x_miss_prem[0]

    p2 = (test_x[1] * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(test_x[1], K_v, tau_v, sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(test_x[1], K_v, tau_v, sigma_v, R_v, D_v))) - test_x_miss_prem[1]

    p3 = (test_x[2] * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(test_x[2], K_v, tau_v, sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(test_x[2], K_v, tau_v, sigma_v, R_v, D_v))) - test_x_miss_prem[2]

    return [p1, p2, p3]
# fsolve algorithm, effective but does not consider bounds
sigma_est, R_est, D_est = (fsolve(price_calc_x, [1., 0.11, 0.7]))
print(f"The volatility is estimated at {sigma_est}, the interest rate at {R_est}, and the dividend at {D_est}")
# least_squares algorithm, does consider bounds
sigma_est, R_est, D_est = (least_squares(price_calc_x, [1., 0.11, 0.4],
                                         bounds = ([1e-14, 1e-14, 0.0],
                                                   [3., .114, .5654]))).get("x")
print(f"The volatility is estimated at {sigma_est}, the interest rate at {R_est}, and the dividend at {D_est}")

least_squares(price_calc_x, [0.8, 0.07, 0.], bounds = ([1e-14, 1e-14, 0.0], [3., .114, .5654])).get("nfev")
fsolve(price_calc_x, [0.8, 0.07, 0.])

# varying K ===========================================

test_K = np.array([670, 700, 730])

# assume we get an initial estimate and return [sigma, R, D]
def price_calc_K(arr):
    """Provides series of 3 equations when K is varied"""

    sigma_v, R_v, D_v = arr

    p1 = (x_v * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(x_v, test_K[0], tau_v, sigma_v, R_v, D_v))
          - test_K[0] * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(x_v, test_K[0], tau_v, sigma_v, R_v, D_v)) -
          test_K_miss_prem[0])

    p2 = (x_v * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(x_v, test_K[1], tau_v, sigma_v, R_v, D_v))
          - test_K[1] * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(x_v, test_K[1], tau_v, sigma_v, R_v, D_v)) -
          test_K_miss_prem[1])

    p3 = (x_v * np.exp(-D_v * tau_v) * norm.cdf(d1_calc(x_v, test_K[2], tau_v, sigma_v, R_v, D_v))
          - test_K[2] * np.exp(-R_v * tau_v) * norm.cdf(d2_calc(x_v, test_K[2], tau_v, sigma_v, R_v, D_v)) -
          test_K_miss_prem[2])

    return [p1, p2, p3]

# fsolve algorithm, effective but does not consider bounds
sigma_est, R_est, D_est = (fsolve(price_calc_K, [0.5, 0.20, 0.09]))
print(f"The volatility is estimated at {sigma_est}, the interest rate at {R_est}, and the dividend at {D_est}")


# least_squares algorithm, does consider bounds
sigma_est, R_est, D_est = (least_squares(price_calc_K, [0.5, 0.20, 0.09],
                                         bounds = ([0.01, 0.01, 0.0],
                                                   [3., 0.9, 0.7]))).get("x")
print(f"The volatility from K is estimated at {sigma_est}, the interest rate at {R_est}, and the dividend at {D_est}")





# varying tau ===========================================

test_tau = np.array([15/252, 43/252, 100/252])

# assume we get an initial estimate and return [sigma, R, D]
def price_calc_tau(arr):
    """Returns a series of 3 equations when tau is varied"""

    sigma_v, R_v, D_v = arr


    p1 = (x_v * np.exp(-D_v * test_tau[0]) * norm.cdf(d1_calc(x_v, K_v, test_tau[0], sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * test_tau[0]) * norm.cdf(d2_calc(x_v, K_v, test_tau[0], sigma_v, R_v, D_v)) -
          test_tau_miss_prem[0])

    p2 = (x_v * np.exp(-D_v * test_tau[1]) * norm.cdf(d1_calc(x_v, K_v, test_tau[1], sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * test_tau[1]) * norm.cdf(d2_calc(x_v, K_v, test_tau[1], sigma_v, R_v, D_v)) -
          test_tau_miss_prem[1])

    p3 = (x_v * np.exp(-D_v * test_tau[2]) * norm.cdf(d1_calc(x_v, K_v, test_tau[2], sigma_v, R_v, D_v))
          - K_v * np.exp(-R_v * test_tau[2]) * norm.cdf(d2_calc(x_v, K_v, test_tau[2], sigma_v, R_v, D_v)) -
          test_tau_miss_prem[2])

    return [p1, p2, p3]

# fsolve algorithm, effective but does not consider bounds
sigma_est, R_est, D_est = (fsolve(price_calc_tau, [0.5, 0.20, 0.09]))
print(f"The volatility is estimated at {sigma_est}, the interest rate at {R_est}, and the dividend at {D_est}")





# variables ===========================================================
# variable selection from Bouchouev
x = 710.82
tau = 43/252
R = 0.05
D = 0.02


# variable selection from me
sigma = 0.25
K = 730


test_x = np.array([670, 710.82, 730])
test_K = np.array([670, 700, 730])
test_tau = np.array([15/252, 43/252, 100/252])
tau_v, K_v, x_v = 43/252, 730, 710.82
test_x_miss_prem = premium_calc(test_x, tau_v, K_v, sigma, R, D)
test_K_miss_prem = premium_calc(x_v, tau_v, test_K, sigma, R, D)
test_tau_miss_prem = premium_calc(x_v, test_tau, K_v, sigma, R, D)

for i in range(3):
    test_x_miss_prem[i] = round(test_x_miss_prem[i], 2) # cap off at 1p difference
    test_K_miss_prem[i] = round(test_K_miss_prem[i], 2) # cap off at 1p difference
    test_tau_miss_prem[i] = round(test_tau_miss_prem[i], 2) # cap off at 1p difference
    # after that, include stochastic disturbance term
    # after that? non-random disturbance term that increases at ends according to smile
    
print(f"For x the call options are {test_x_miss_prem}")
print(f"For K the call options are {test_K_miss_prem}")
print(f"For tau the call options are {test_tau_miss_prem}")


# actual optimisation calls ====================================================

# for x
sigma_x_est, R_x_est, D_x_est = (least_squares(price_calc_x, [2., 5., 10.],
                                         bounds = ([0.01, 0.01, 0.0],
                                                   [3., 11.6, 30.]))).get("x")
print(f"The volatility from x is estimated at {sigma_x_est}, the interest rate at {R_x_est}, and the dividend at {D_x_est}")

# for K
sigma_K_est, R_K_est, D_K_est = (least_squares(price_calc_K, [2., 5., 10.],
                                         bounds = ([0.01, 0.01, 0.0],
                                                   [3., 11.6, 30.]))).get("x")
print(f"The volatility from K is estimated at {sigma_K_est}, the interest rate at {R_K_est}, and the dividend at {D_K_est}")

# for tau
sigma_tau_est, R_tau_est, D_tau_est = (least_squares(price_calc_tau, [2., 5., 10.],
                                         bounds = ([0.01, 0.01, 0.0],
                                                   [3., 11.6, 30.]))).get("x")
print(f"The volatility from K is estimated at {sigma_tau_est}, the interest rate at {R_tau_est}, and the dividend at {D_tau_est}")

# original 0.5, 0.20, 0.09
# highest dividend paying stock is 30%
# highest LIBOR rate was around 11% https://www.macrotrends.net/1433/historical-libor-rates-chart

labels = [
    "True Value",
    "Calculated from x",
    "Calculated from K",
    "Calculated from maturity"
]

sigma_values = [
    sigma,
    sigma_x_est,
    sigma_K_est,
    sigma_tau_est
]

R_values = [
    R,
    R_x_est,
    R_K_est,
    R_tau_est
]

D_values = [
    D,
    D_x_est,
    D_K_est,
    D_tau_est
]

positions = np.arange (len(labels))
# figsize=(15,5)
fig, axes = plt.subplots(3,1)

axes[0].scatter(positions, sigma_values, s=70)
axes[0].axhline(
    sigma,
    linestyle="--",
    label="Original value",
)
axes[0].set_title(r"Estimates of $\sigma$")
axes[0].set_ylabel(r"$\sigma$")
axes[0].set_xticks(positions, labels, rotation=30, ha="right")
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend()

# Interest-rate estimates
axes[1].scatter(positions, R_values, s=70)
axes[1].axhline(
    R,
    linestyle="--",
    label="Original value",
)
axes[1].set_title(r"Estimates of $R$")
axes[1].set_ylabel(r"$R$")
axes[1].set_xticks(positions, labels, rotation=30, ha="right")
axes[1].grid(axis="y", alpha=0.3)
axes[1].legend()

# Dividend-yield estimates
axes[2].scatter(positions, D_values, s=70)
axes[2].axhline(
    D,
    linestyle="--",
    label="Original value",
)
axes[2].set_title(r"Estimates of $D$")
axes[2].set_ylabel(r"$D$")
axes[2].set_xticks(positions, labels, rotation=30, ha="right")
axes[2].grid(axis="y", alpha=0.3)
axes[2].legend()

fig.suptitle("Original and recovered Black–Scholes parameters")
fig.tight_layout(h_pad=10)

plt.show()




# Stochastic disturbance ===========================================================
# variable selection from Bouchouev
x = 710.82
tau = 43/252
R = 0.05
D = 0.02


# variable selection from me
sigma = 0.25
K = 730
np.random.seed(123)

test_x = np.array([670, 710.82, 730])
test_K = np.array([670, 700, 730])
test_tau = np.array([15/252, 43/252, 100/252])
tau_v, K_v, x_v = 43/252, 730, 710.82
test_x_miss_prem = premium_calc(test_x, tau_v, K_v, sigma, R, D)
test_K_miss_prem = premium_calc(x_v, tau_v, test_K, sigma, R, D)
test_tau_miss_prem = premium_calc(x_v, test_tau, K_v, sigma, R, D)

# this setup will successfully instantiate n_tests amount of stochastically disturbed option premiums
# premiums all come from the same underlying distribution, at this point disturbance is random

n_tests = 1000
dist_mu = 0
dist_sd= 1


# disturbance process ====================================================

half_n = n_tests // 2

# x
# test_x_set = test_x_miss_prem * np.ones((n_tests,3))

# disturbed_x_set = test_x_set + np.random.normal(dist_mu, dist_sd, (n_tests,3))

disturbances = np.random.normal (dist_mu, dist_sd, (half_n, 3))
disturbed_x_set = np.vstack([test_x_miss_prem + disturbances, test_x_miss_prem - disturbances,])

for i, set in enumerate(disturbed_x_set):
    for j, val in enumerate (set):
        set[j] = round(set[j],2)
    
# K
# test_K_set = test_K_miss_prem * np.ones((n_tests,3))

# disturbed_K_set = test_K_set + np.random.normal(dist_mu, dist_sd, (n_tests,3))

disturbances = np.random.normal (dist_mu, dist_sd, (half_n, 3))
disturbed_K_set = np.vstack([test_K_miss_prem + disturbances, test_K_miss_prem - disturbances,])

for i, set in enumerate(disturbed_K_set):
    for j, val in enumerate (set):
        set[j] = round(set[j],2)


# tau        
# test_tau_set = test_tau_miss_prem * np.ones((n_tests,3))

# disturbed_tau_set = test_tau_set + np.random.normal(dist_mu, dist_sd, (n_tests,3))

disturbances = np.random.normal (dist_mu, dist_sd, (half_n, 3))
disturbed_tau_set = np.vstack([test_tau_miss_prem + disturbances, test_tau_miss_prem - disturbances,])

for i, set in enumerate(disturbed_tau_set):
    for j, val in enumerate (set):
        set[j] = round(set[j],2)
        


# actual optimisation calls ====================================================
MC_x_inversions = []
MC_K_inversions = []
MC_tau_inversions = []

holder_x = test_x_miss_prem   # preserve original premiums
holder_K = test_K_miss_prem
holder_tau = test_tau_miss_prem

# MC x inversions
for set in disturbed_x_set:

    test_x_miss_prem = set      # overwrite with disturbed premiums

    # specific case inversions
    MC_x_inversions.append((least_squares(price_calc_x, [0.5, 0.06, 0.01],
                                     bounds = ([1e-14, 1e-14, 0.0], [3., .09, .5653]))).get("x"))
    
test_x_miss_prem = holder_x   # overwrite with original premium

MC_x_inversions


# MC K inversions
for set in disturbed_K_set:

    test_K_miss_prem = set      # overwrite with disturbed premiums

    # specific case inversions
    MC_K_inversions.append((least_squares(price_calc_K, [0.5, 0.06, 0.01],
                                     bounds = ([1e-14, 1e-14, 0.0], [3., .09, .5653]))).get("x"))
    
test_K_miss_prem = holder_K   # overwrite with original premium

MC_K_inversions



# MC tau inversions
for set in disturbed_tau_set:

    test_tau_miss_prem = set      # overwrite with disturbed premiums

    # specific case inversions
    MC_tau_inversions.append((least_squares(price_calc_tau, [0.5, 0.06, 0.01],
                                     bounds = ([1e-14, 1e-14, 0.0], [3., .09, .5653]))).get("x"))
    
test_tau_miss_prem = holder_tau   # overwrite with original premium

MC_tau_inversions




# Moment Calculation ====================================================

MC_x_inversions = np.array(MC_x_inversions)
MC_K_inversions = np.array(MC_K_inversions)
MC_tau_inversions = np.array(MC_tau_inversions)



# x 
total_x_volatility = []
total_x_R = []
total_x_D = []

for set in MC_x_inversions:

    total_x_volatility.append(set[0])
    total_x_R.append(set[1])
    total_x_D.append(set[2])

total_x_volatility = np.array(total_x_volatility, dtype=np.float64)
total_x_R = np.array(total_x_R, dtype=np.float64)
total_x_D = np.array(total_x_D, dtype=np.float64)

sigma_x_est, R_x_est, D_x_est = np.mean(total_x_volatility), np.mean(total_x_R), np.mean(total_x_D)
sigma_x_sd, R_x_sd, D_x_sd = np.std(total_x_volatility), np.std(total_x_R), np.std(total_x_D)



# K

total_K_volatility = []
total_K_R = []
total_K_D = []

for set in MC_K_inversions:

    total_K_volatility.append(set[0])
    total_K_R.append(set[1])
    total_K_D.append(set[2])

total_K_volatility = np.array(total_K_volatility, dtype=np.float64)
total_K_R = np.array(total_K_R, dtype=np.float64)
total_K_D = np.array(total_K_D, dtype=np.float64)
    
sigma_K_est, R_K_est, D_K_est = np.mean(total_K_volatility), np.mean(total_K_R), np.mean(total_K_D)
sigma_K_sd, R_K_sd, D_K_sd = np.std(total_K_volatility), np.std(total_K_R), np.std(total_K_D)



# tau

total_tau_volatility = []
total_tau_R = []
total_tau_D = []

for set in MC_tau_inversions:

    total_tau_volatility.append(set[0])
    total_tau_R.append(set[1])
    total_tau_D.append(set[2])

total_tau_volatility = np.array(total_tau_volatility, dtype=np.float64)
total_tau_R = np.array(total_tau_R, dtype=np.float64)
total_tau_D = np.array(total_tau_D, dtype=np.float64)


sigma_tau_est, R_tau_est, D_tau_est = np.mean(total_tau_volatility), np.mean(total_tau_R), np.mean(total_tau_D)
sigma_tau_sd, R_tau_sd, D_tau_sd = np.std(total_tau_volatility), np.std(total_tau_R), np.std(total_tau_D)




print(f"The volatility from x is estimated at {sigma_x_est:.5f}, the interest rate at {R_x_est:.5f}, and the dividend at {D_x_est:.5f}")


print(f"The volatility from K is estimated at {sigma_K_est:.5f}, the interest rate at {R_K_est:.5f}, and the dividend at {D_K_est:.5f}")

print(f"The volatility from tau is estimated at {sigma_tau_est:.5f}, the interest rate at {R_tau_est:.5f}, and the dividend at {D_tau_est:.5f}")


print(f"The volatility from x has variance of estimates at {(sigma_x_sd):.5f}, the intersd rate at {(R_x_sd):.5f}, and the dividend at {(D_x_sd):.5f}")


print(f"The volatility from K has variance of estimates at {(sigma_K_sd):.5f}, the intersd rate at {(R_K_sd):.5f}, and the dividend at {(D_K_sd):.5f}")

print(f"The volatility from tau has variance of estimates at {(sigma_tau_sd):.5f}, the intersd rate at {(R_tau_sd):.5f}, and the dividend at {(D_tau_sd):.5f}")


# total_input_invert format

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.hist(total_x_volatility)

plt.show()




#layout='constrained'
fig, axes = plt.subplots(3,3)

# x recoveries

# volatility
axes[0,0].hist(total_x_volatility)
axes[0,0].set_title(r"Volatility")
axes[0,0].set_xlabel(r"Volatility $\sigma$")

# R
axes[0,1].hist(total_x_R)
axes[0,1].set_title(r"Interest rate under $x$")
axes[0,1].set_xlabel(r"Interest Rate $R$")

# D
axes[0,2].hist(total_x_D)
axes[0,2].set_title(r"Dividend rate under $x$")
axes[0,2].set_xlabel(r"Dividend Rate $D$")


# K recoveries

# volatility
axes[1,0].hist(total_K_volatility)
axes[1,0].set_title(r"Volatility under $K$")
axes[1,0].set_xlabel(r"Volatility $\sigma$")

# R
axes[1,1].hist(total_K_R)
axes[1,1].set_title(r"Interest rate under $K$")
axes[1,1].set_xlabel(r"Interest Rate $R$")

# D
axes[1,2].hist(total_K_D)
axes[1,2].set_title(r"Dividend rate under $K$")
axes[1,2].set_xlabel(r"Dividend Rate $D$")


# tau recoveries

# volatility
axes[2,0].hist(total_tau_volatility)
axes[2,0].set_title(r"Volatility under $T-t$")
axes[2,0].set_xlabel(r"Volatility $\sigma$")

# R
axes[2,1].hist(total_tau_R)
axes[2,1].set_title(r"Interest rate under $T-t$")
axes[2,1].set_xlabel(r"Interest Rate $R$")

# D
axes[2,2].hist(total_tau_D)
axes[2,2].set_title(r"Dividend rate under $T-t$")
axes[2,2].set_xlabel(r"Dividend Rate $D$")

fig.suptitle("Recovered stochastically disturbed Black-Scholes parameter distributions")
fig.tight_layout(h_pad=10)
plt.show()



fig, axes = plt.subplots(3, 3, figsize=(12, 9))

hist_data = [ [ total_x_volatility, total_x_R, total_x_D ],
              [ total_K_volatility, total_K_R, total_K_D ],
              [ total_tau_volatility, total_tau_R, total_tau_D ]]

col_titles = [ "Volatility", "Interest Rate", "Dividend Rate" ]
row_titles = [ r"Under $x$", r"Under $K$", "Under $T-t$" ]
x_labs = [ r"Volatility $\sigma$", r"Interest Rate $R$", r"Dividend Rate $D$" ]


for row in range (3):
    for col in range (3):
        axes[row,col].hist(hist_data[row][col])
        axes[row,col].set_xlabel(x_labs[col])

        if row == 0:
            axes[row,col].set_title(col_titles[col])

        if col == 0:
            axes[row,col].set_ylabel(row_titles[row], rotation=90,
                                     labelpad=15, fontsize=11)

fig.suptitle("Recovered stochastically disturbed Black-Scholes parameter distributions")
fig.tight_layout(rect=[0,0,1,.95])
plt.show()





disturbed_sets = [
    disturbed_x_set,
    disturbed_K_set,
    disturbed_tau_set,
]

inversion_sets = [
    MC_x_inversions,
    MC_K_inversions,
    MC_tau_inversions,
]

row_labels = [
    r"Under $x$",
    r"Under $K$",
    r"Under $\tau$",
]

column_titles = [
    r"Recovered volatility $\sigma$",
    r"Recovered interest rate $R$",
    r"Recovered dividend rate $D$",
]

premium_labels = {
    "x": [
        rf"$C(x={test_x[0]:g})$",
        rf"$C(x={test_x[1]:g})$",
        rf"$C(x={test_x[2]:g})$",
    ],
    "K": [
        rf"$C(K={test_K[0]:g})$",
        rf"$C(K={test_K[1]:g})$",
        rf"$C(K={test_K[2]:g})$",
    ],
    "tau": [
        rf"$C(\tau={test_tau[0]:g})$",
        rf"$C(\tau={test_tau[1]:g})$",
        rf"$C(\tau={test_tau[2]:g})$",
    ],
}

all_premium_labels = [
    premium_labels["x"],
    premium_labels["K"],
    premium_labels["tau"],
]


original_premium_sets = [
    test_x_miss_prem,
    test_K_miss_prem,
    test_tau_miss_prem,
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 12),
)

for row, (
    disturbed,
    original,
    inversions,
    labels,
) in enumerate(
    zip(
        disturbed_sets,
        original_premium_sets,
        inversion_sets,
        all_premium_labels,
    )
):
    premium_errors = disturbed - original[None, :]

    for parameter_index in range(3):
        ax = axes[row, parameter_index]

        for premium_index in range(3):
            ax.scatter(
                premium_errors[:, premium_index],
                inversions[:, parameter_index],
                alpha=0.6,
                s=20,
                label=labels[premium_index],
            )

        ax.axvline(0, linestyle="--", linewidth=1)

        if row == 0:
            ax.set_title(column_titles[parameter_index])

        if parameter_index == 0:
            ax.set_ylabel(row_labels[row])

        ax.set_xlabel(r"Premium disturbance $\widetilde C-C$")
        ax.grid(alpha=0.3)

        if parameter_index == 2:
            ax.legend(fontsize=8)

fig.suptitle(
    "Recovered parameters against stochastic premium disturbances",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 12),
)


markers = ["o", "s", "^"]
line_styles = ["-", "--", ":"]

for premium_index in range(3):
    x_data = premium_errors[:, premium_index]
    y_data = inversions[:, parameter_index]

    # Display only every fifth observation to reduce clutter
    display_indices = np.arange(0, len(x_data), 5)

    ax.scatter(
        x_data[display_indices],
        y_data[display_indices],
        marker=markers[premium_index],
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        s=24,
        label=labels[premium_index],
    )

    # Linear trend calculated from all observations
    coefficients = np.polyfit(x_data, y_data, deg=1)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = np.polyval(coefficients, x_line)

    ax.plot(
        x_line,
        y_line,
        linestyle=line_styles[premium_index],
        linewidth=1.2,
        color="black",
    )
fig.suptitle(
    "Recovered parameters against stochastic premium disturbances",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()















rng = np.random.default_rng(42)

markers = ["o", "s", "^"]
line_styles = ["-", "--", ":"]


original_premium_sets = [
    test_x_miss_prem,
    test_K_miss_prem,
    test_tau_miss_prem,
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 12),
)

for row, (
    disturbed,
    original,
    inversions,
    labels,
) in enumerate(
    zip(
        disturbed_sets,
        original_premium_sets,
        inversion_sets,
        all_premium_labels,
    )
):
    premium_errors = disturbed - original[None, :]

    for parameter_index in range(3):
        ax = axes[row, parameter_index]

        for premium_index in range(3):
            x_data = premium_errors[:, premium_index]
            y_data = inversions[:, parameter_index]

            n_display = min(200, len(x_data))
            display_indices = rng.choice(
                len(x_data),
                size=n_display,
                replace=False,
            )

            ax.scatter(
                x_data[display_indices],
                y_data[display_indices],
                marker=markers[premium_index],
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
                s=22,
                label=labels[premium_index],
            )

            coefficients = np.polyfit(x_data, y_data, 1)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)

            ax.plot(
                x_line,
                np.polyval(coefficients, x_line),
                color="black",
                linestyle=line_styles[premium_index],
                linewidth=1.2,
            )

            if row == 0:
                ax.set_title(column_titles[parameter_index])

            if parameter_index == 0:
                ax.set_ylabel(row_labels[row])

            ax.set_xlabel(r"Premium disturbance $\widetilde C-C$")
            ax.grid(alpha=0.3)

            if parameter_index == 2:
                ax.legend(fontsize=8)


            
fig.suptitle(
    "Recovered parameters against stochastic premium disturbances",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


rng = np.random.default_rng(42)

np.random.seed(1234)

original_premium_sets = [
    test_x_miss_prem,
    test_K_miss_prem,
    test_tau_miss_prem,
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 12),
)

for row, (
    disturbed,
    original,
    inversions,
    labels,
) in enumerate(
    zip(
        disturbed_sets,
        original_premium_sets,
        inversion_sets,
        all_premium_labels,
    )
):
    premium_errors = disturbed - original[None, :]

    for parameter_index in range(3):
        ax = axes[row, parameter_index]

        for premium_index in range(3):
            x_data = premium_errors[:, premium_index]
            y_data = inversions[:, parameter_index]

            n_display = min(200, len(x_data))
            display_indices = rng.choice(
                len(x_data),
                size=n_display,
                replace=False,
            )

            ax.scatter(
                x_data[display_indices],
                y_data[display_indices],
                # marker=markers[premium_index],
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
                s=22,
                # label=labels[premium_index],
            )

            # I want a couple lines over each true value
            if parameter_index == 0:
                tru_param_val = sigma
            elif parameter_index == 1:
                tru_param_val = R
            elif parameter_index == 2:
                tru_param_val = D
            else:
                print("fuck")
                
            ax.axhline(
                tru_param_val,
                linestyle="--",
                # label="Original value",
            )

            if row == 0:
                ax.set_title(column_titles[parameter_index])

            if parameter_index == 0:
                ax.set_ylabel(row_labels[row])

            ax.set_xlabel(r"Premium disturbance $\widetilde C-C$")
            ax.grid(alpha=0.3)

            if parameter_index == 2:
                ax.legend(fontsize=8)


            
fig.suptitle(
    "Recovered parameters against stochastic premium disturbances",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()






total_x_mu = total_x_R - total_x_D
total_K_mu = total_K_R - total_K_D
total_tau_mu = total_tau_R - total_tau_D

fig, axes = plt.subplots(1,3, figsize=(10,4))

axes[0].hist(total_x_mu)
axes[0].axvline(0.03, color='k', linestyle='dashed', linewidth=1)
axes[0].set_title(r"Estimate from $x$")
axes[0].set_ylabel("Frequency")
axes[0].set_xlabel("Drift $\mu$")

axes[1].hist(total_K_mu)
axes[1].axvline(0.03, color='k', linestyle='dashed', linewidth=1)
axes[1].set_title(r"Estimate from $K$")
axes[1].set_ylabel("Frequency")
axes[1].set_xlabel("Drift $\mu$")

axes[2].hist(total_tau_mu)
axes[2].axvline(0.03, color='k', linestyle='dashed', linewidth=1)
axes[2].set_title(r"Estimate from $T-t$")
axes[2].set_ylabel("Frequency")
axes[2].set_xlabel("Drift $\mu$")

fig.suptitle("Recovered drift term from stochastically disturbed premiums")
fig.tight_layout(h_pad=10)

plt.show()


help(plt.pl)

