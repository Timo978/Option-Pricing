import sympy as sp
from sympy.stats import Normal, cdf
from py_vollib_vectorized import vectorized_black_scholes
import numpy as np
import matplotlib.pyplot as plt

S, K, sigma, ttm, r, d1, d2 = sp.symbols('S,K,sigma,ttm,r,d_1,d_2')

    # define a symbol to represent the normal CDF
N = sp.Function('N')
    # Black76 price in deri
C = sp.exp(-r * ttm) * N(d2)
# P = N(-d2) * K - F * N(-d1)

    # expanded d1 and d2 for substitution:
d1_sub = (sp.ln(S / K) + ((sp.Rational(1, 2) * sigma ** 2) + r) * ttm) / (sigma * sp.sqrt(ttm))
d2_sub = (sp.ln(S / K) + (r - (sp.Rational(1, 2) * sigma ** 2)) * ttm) / (sigma * sp.sqrt(ttm))

# instance a standard normal distribution:
Norm = Normal('N', 0.0, 1.0)

# define the long form b-s equation with all substitutions:
bs_c = C.subs(N, cdf(Norm)).subs(d2, d2_sub).subs(d1, d1_sub)

# Callable function for black and scholes price:
# example usage: bs_c_calc(100, 98, 0.15, 0.38)
bs_c_calc = sp.lambdify((S, K, sigma, ttm, r), bs_c)

delta_c = sp.simplify(sp.diff(bs_c, S))
delta_c_calc = sp.lambdify((S, K, sigma, ttm, r), delta_c)

bs_c_calc(100, 50, 0.15, 0.38,0.04)
delta_c_calc(100, 100, 0.15, 0.38,0.04)

dt = []
for i in np.arange(98,102,0.02):
    dt.append(delta_c_calc(100, i, 0.3, 0.1, 0.05))

plt.figure(figsize=(5,5))
plt.plot(np.arange(98,102,0.02),dt)
plt.show()