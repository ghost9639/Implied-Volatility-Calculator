
# Table of Contents

1.  [Project](#org79e24b9)
2.  [Theoretical background](#org90050a9)
3.  [Algorithm](#org064646e)
4.  [Simulated data inversion](#org9d0f4db)
5.  [Real data inversion](#org64f5abd)
6.  [Appendix](#org5c2bb72)



<a id="org79e24b9"></a>

# Project

This project reproduces major advancements in option pricing analytics, chiefly targeting the recovery of the implied local volatility for European options. This culminates in an inversion on real options data using a well-posed algorithm. The report also contains derivations for key mathematical results, and an in-depth technical explanation on the algorithm used. Full implementation in Python, using a wide range of NumPy tools.

This project includes:

1.  dense theoretical derivations for mathematical theory,
2.  analysis of Black-Scholes model solution,
3.  detailed explanation of well-posed algorithm,
4.  implementation of (Lishang, Jiang and Qihong, Chen and Ligun, Wang and Jin E., Zhang, 2003) algorithm in Python.


<a id="org90050a9"></a>

# Theoretical background

The Black-Scholes equation underpins many industry standard methods of accurately pricing options. The parameters of the Black-Scholes equation for specific markets pose great value to practitioners. The Black-Scholes inverse problem concerns methods of effectively recovering these parameters from real markets. The Black-Scholes partial differential equation:

$\frac{\partial u}{\partial t} + &sigma; x<sup>2</sup> \frac{\partial^2 u}{\partial x^2} + (R - D) x \frac{\partial u}{\partial  x} - R u = 0,$

where $u$ is the call option premium, $t$ is the current time, $\sigma$ is the underlying market volatility, $x$ is the price of an option, $R$ is the interest rate and $D$ is the dividend rate, sets the arbitrage-free price of an option in classical option pricing. Recovering &sigma; permits the arbitrage-free pricing and evaluation of existing options.


<a id="org064646e"></a>

# Algorithm

The algorithm converts the Black-Scholes equation to an optimal control problem. First, we can numerically solve for the theoretical value of the option premium under an initial volatility estimate. Then, we solve an adjoint equation testing the goodness of fit between the theoretical and obtained curve. Finally, a variation inequality solver estimates the numerical value of the fit over a false interval, returning an improved estimate for volatility. These steps are repeated until convergence. [Project file](Programs/LishangAlg.py) available in /Programs/LishangAlg.py.


<a id="org9d0f4db"></a>

# Simulated data inversion

I simulated data according to the specifications: $D = 0$, $R = 0.12$, $a_0 = 0.005$, $a_1 = 0.045$, $\eta \leq 10^{-4}$, $x^* = 10$, $\tau^*=1$,

and the cases:

1.  'flat' volatility,
    $&sigma;<sub>p</sub>(x) = 0.25,$
2.  'smile' volatility,
    $&sigma;<sub>p</sub>(x) = (ln (s / 10))<sup>2</sup> / 40 + 0.2,$
3.  'skew' volatility,
    $&sigma;<sub>p</sub>(x) = -(ln(s/10))<sup>3</sup> / 80 + 0.2,$

For the optimal control problem, I set $L = M = 10$, $N = 0.01$, $\eta = 0.0001$, $\alpha = 0.45$, 5,000 loops within the variational inequality solver, and a maximum of 600 iterations for the overall function. The results showed a high degree of accuracy, and the inverted local volatility maintained the correct shape for the underlying volatility with a minor bias.

![img](Report/images/flatpred.svg "Inverted volatility for 'flat' case")

![img](Report/images/skewpred.svg "Inverted volatilty in 'skewed' case")

![img](Report/images/smilepred.svg "Inverted volatility in 'smile' case")


<a id="org64f5abd"></a>

# Real data inversion

Using the Bloomberg Excel API, I got a wide range of strike prices for European call options on Allianz (ALV GY Equity), a major insurance company with a high volume of trading activity in European markets. After discarding repeated and far out of the money options, I obtained the local volatility. Repricing revealed a high degree of accuracy for in the money options.

![img](Report/images/diagnostics_strike.svg "Inverted local vs Bloomberg implied volatility")

![img](Report/images/diagnostic5.svg "Repriced options vs true market value")


<a id="org5c2bb72"></a>

# Appendix

Lishang, Jiang and Qihong, Chen and Ligun, Wang and Jin E., Zhang (2003). *A New Well-Posed Algorithm to Recover Implied Local Volatility*, {Quantitative Finance}.

