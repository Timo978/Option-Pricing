import numpy as np
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_black_scholes_merton

S0 = 30  ### 股价 30
r = 0.03
q = 0.01
k = 0.3
V0 = 1
rho = -0.7
theta = 0.5
xi = 0.5
B = S0 * 1.1  ### 障碍 33
T = 1.0
F0 = S0 * np.exp((r - q) * T)
npath = 10000
nstep = 100
dt = T / nstep
tgrid = np.linspace(0, T, nstep + 1)
CH = np.array([[1., rho], [rho, 1.]])
L = np.linalg.cholesky(CH)
S = S0 * np.ones((nstep + 1, npath))
F = F0 * np.ones((nstep + 1, npath))
V = V0 * np.ones((nstep + 1, npath))
I = np.ones((1, npath))  ### Indicator

for i in range(0, nstep):
    ZH = np.random.normal(size=(2, npath // 2))
    ZA = np.c_[ZH, -ZH]  # antithetic sampling
    Z = L @ ZA
    dS = r * S[i, :] * dt + np.sqrt(V[i, :]) * S[i, :] * np.sqrt(dt) * Z[0, :]
    S[i + 1, :] = S[i, :] + dS
    dV = k * (theta - V[i, :]) * dt + xi * np.sqrt(V[i, :]) * np.sqrt(dt) * Z[1, :]
    V[i + 1, :] = np.maximum(V[i, :] + dV, 0)
    St = F[i + 1, :] * np.exp(-(r - q) * (T - tgrid[i + 1]))
    I = I * (St < B)

ST = F[-1, :]

# Up-and-Out Call
K = np.linspace(0.8 * S0, B, 10)
UOCheston = np.zeros(K.shape)
for i in range(0, K.size):
    UOCheston[i] = np.exp(-r * T) * np.mean(np.maximum(ST - K[i], 0) * I)

plt.plot(K, UOCheston, 'o-', label='Heston')
plt.plot(K, vectorized_black_scholes_merton('c', S0,K, T, r,np.mean(V), q,return_as='array'), '--b', label='stable vol')
plt.legend()
plt.title('Barrier: Heston')
plt.show()

