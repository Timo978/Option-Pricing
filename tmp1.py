import numpy as np
import scipy as sp

S = 6500  # underlying spot
K = 6500  # strike
KI_Barrier = 0.75  # down in barrier of put
KO_Barrier = 1.03  # autocall barrier
KO_Coupon = 0.25  # autocall coupon (p.a.)
Bonus_Coupon = 0.25  # bonus coupon (p.a.)
r = 0.03  # risk-free interest rate
div = 0.01  # dividend rate
T = 1  # time to maturity in years
v = 0.12  # volatility
N = 252 * T  # number of discrete time points for whole tenor
dt = T / N  # delta t
simulations = 30000

price_trajectories = []
for i in range(simulations):
    dZ = sp.random.normal(0, 1, N)

    stockprices = np.cumprod(np.exp((r - div -0.5 * v ** 2) * dt + v * np.sqrt(dt) * dZ)) * S

    # check if Knockout, monthly observation
    n = int(1.0 / dt / 12.0)  # number of time points in every month
    s = slice((n - 1)*3, N, n)
    stockprices_slice = stockprices[s]
    if stockprices_slice.max() >= KO_Barrier:
        idx = np.argmax(stockprices_slice >= KO_Barrier)
        time_to_KO = (idx + 1) / 12.0
        pv = (KO_Coupon * time_to_KO + 1) * np.exp(-r * time_to_KO)
        price_trajectories.append(pv)
        continue

    # if no KO, bonus coupon or down in put
    indicator_KI = stockprices.min() <= KI_Barrier
    pv = ((1 - indicator_KI) * (Bonus_Coupon * T + 1) \
          + indicator_KI * (stockprices[-1] - K + 1)) * np.exp(-r * T)
    price_trajectories.append(pv)

option_price = np.sum(price_trajectories) / simulations