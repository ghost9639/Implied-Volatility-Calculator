import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# from matplotlib.colors import LightSource
# from matplotlib import cm
# import seaborn as sns


# variable selection from Bouchouev
x = 710.82
tau = 43/252
R = 0.05
D = 0.02


# variable selection from me
sigma = 0.25
K = 730
steps = 10000

# as x increases
path = np.linspace(1e-14, x+300, steps)
u = np.zeros((3,steps), dtype=np.float64)
set_K = [670, 700, 730]
set_asymptote = np.zeros((3,steps), dtype=np.float64)

for i in range(0,3):

    d1 = (np.log(path/set_K[i]) + (R - D + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
    d2 = (np.log(path/set_K[i]) + (R - D - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))

    u[i,:] = (path * np.exp(-D * tau) * norm.cdf(d1) - set_K[i] * np.exp(-R * tau) * norm.cdf(d2))

    set_asymptote[i,:] = (path * np.exp(-D * tau) - set_K[i] * np.exp(-R*tau))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

major_yticks = [0,100,200,300]
major_xticks = np.arange(500,1100,100)

ax.set_xticks(major_xticks)
ax.set_yticks(major_yticks)

plt.plot(path,u[0,:],label=r"Black-Scholes Price, $\; K = 670$", ls = '-')
plt.plot(path,set_asymptote[0,:],label=r"$x e^{-D \tau} - K e^{-R \tau}, \; K = 670$", ls = '--')

plt.plot(path,u[1,:],label=r"Black-Scholes Price, $\; K = 700$", ls = '-.')
plt.plot(path,set_asymptote[1,:],label=r"$x e^{-D \tau} - K e^{-R \tau}, \; K = 700$", ls = ':')

plt.plot(path,u[2,:],label=r"Black-Scholes Price, $\; K = 730$", ls = (0, (3, 10, 1, 10, 1, 10)))
plt.plot(path,set_asymptote[2,:],label=r"$x e^{-D \tau} - K e^{-R \tau}, \; K = 730$", ls = (0, (3, 1, 1, 1)))
plt.title("Value of option premium across initial stock price")
plt.xlabel(r"Stock Price $x$")
plt.ylabel(r"Option Premium $u(x,0)$")

plt.ylim(0,300)
plt.legend(loc='upper left')
plt.xlim(550,1000)
plt.show()





# as tau increases
set_sigma = [0.1, 0.25, 0.5]
path = np.linspace(1e-14, 2, 252*2)     # trading days in a year

u = np.zeros((3, 252*2), dtype=np.float64)

for i in range (0,3):

    d1 = (np.log(x/K) + (R - D + 0.5 * (set_sigma[i]**2)) * path) / (set_sigma[i] * np.sqrt(path))
    d2 = (np.log(x/K) + (R - D - 0.5 * (set_sigma[i]**2)) * path) / (set_sigma[i] * np.sqrt(path))

    u[i,:] = (x * np.exp(-D * path) * norm.cdf(d1) - K * np.exp(-R * path) * norm.cdf(d2))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

major_yticks = np.arange(0,250,50)
major_xticks = np.arange(0,2.5,0.5)

ax.set_xticks(major_xticks)
ax.set_yticks(major_yticks)

ax.set_ylim(-0.1,202)
ax.set_xlim(-0.001, 2.02)

plt.plot(path,u[2,:], label=r"$\sigma$ = 0.5")
plt.plot(path,u[1,:], label=r"$\sigma$ = 0.25")
plt.plot(path,u[0,:], label=r"$\sigma$ = 0.1")
plt.title("Value of option premium as time till maturity changes")
plt.xlabel(r"Time till maturity $T-t$")
plt.ylabel("Option Premium $u(x,0)$")
plt.legend(loc="upper left")
plt.show()





from mpl_toolkits.mplot3d import axes3d

# as x and K both vary
x_steps = 1000
K_steps = 500

x_path = np.linspace(5, x+300, x_steps)
K_path = np.linspace(x-100, x+100, K_steps)

X_mesh, K_mesh = np.meshgrid(x_path, K_path, indexing="ij")

d1 = (np.log(X_mesh/K_mesh) + (R - D + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
d2 = (np.log(X_mesh/K_mesh) + (R - D - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))

prices = (X_mesh * np.exp(-D * tau) * norm.cdf(d1) - K_mesh * np.exp(-R * tau) * norm.cdf(d2))

x_ticker = np.arange(500,900,100)
y_ticker = np.arange(-200,1200,200)
z_ticker = np.arange(-100,500,100)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121, projection='3d')
# surf = ax.plot_surface(K_mesh, X_mesh, prices, cmap = "viridis")
ax.plot_wireframe(K_mesh, X_mesh, prices, rstride=40, cstride=40) # not bad but doesn't communicate altitude 

ax.set_xlabel(r"Strike Price $K$")
ax.set_ylabel(r"Option Price $x$")
ax.set_zlabel(r"Call Option Premium $u(x,0)$")
ax.set_xticks(x_ticker)
ax.set_xlim(600,825)

ax.set_yticks(y_ticker)
ax.set_ylim(0,1000)

ax.set_zticks(z_ticker)
ax.set_zlim(0,400)

plt.title("Premium across strike and option prices")
plt.show()





# inverse problem ===================================

# as sigma increases - is this valuable?
path = np.linspace(1e-14, 1, 100)     

d1 = (np.log(x/K) + (R - D + 0.5 * (path**2)) * tau) / (path * np.sqrt(tau))
d2 = (np.log(x/K) + (R - D - 0.5 * (path**2)) * tau) / (path * np.sqrt(tau))

u = (x * np.exp(-D * tau) * norm.cdf(d1) - K * np.exp(-R * tau) * norm.cdf(d2))

plt.plot(path,u)
plt.title("Value of the option premium as volatility increases")
plt.xlabel("Volatility (σ)")
plt.ylabel("Option Premium u(x,0)")
plt.show()



# as R and D vary
R_steps = 100
D_steps = 100

R_path = np.linspace(0, 1, R_steps)
D_path = np.linspace(0, 1, D_steps)

R_mesh, D_mesh = np.meshgrid(R_path, D_path, indexing="ij")

d1 = (np.log(x/K) + (R_mesh - D_mesh + 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))
d2 = (np.log(x/K) + (R_mesh - D_mesh - 0.5 * (sigma**2)) * tau) / (sigma * np.sqrt(tau))

prices = (x * np.exp(-D_mesh * tau) * norm.cdf(d1) - K * np.exp(-R_mesh * tau) * norm.cdf(d2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(R_mesh, D_mesh, prices, cmap = "viridis")
ax.set_xlabel(r"Risk-Free Rate $R$")
ax.set_xlim(1,0)
ax.set_ylabel(r"Dividend Percentage $D$")
ax.set_ylim(1,0)
ax.set_zlabel(r"Call Option Premium $u(x,0)$")
plt.title("Premium across RFR and Dividend")
plt.show()
