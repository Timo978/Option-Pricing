import numpy as np

M = 5000
S = 276.10
r = 0.16/100
T = 58 / 365
v = 0.407530933
K = 230
S_max = 500
delta_T = T/M
delta_S = S_max/M

f_matrx = np.zeros([M+1,M+1])
f_matrx[:,0] = 0.0
for i in range(M + 1):
    f_matrx[M, i] = float(max(delta_S * i - K, 0))
    # 边界条件③：S=S_max的时候，call=S_max-K
f_matrx[:, M] = float(S_max - K)

def calculate_coeff(j):

    vj2 = (v * j)**2
    aj = 0.5 * delta_T * (r * j - vj2)
    bj = 1 + delta_T * (vj2 + r)
    cj = -0.5 * delta_T * (r * j + vj2)
    return aj, bj, cj

matrx = np.zeros([M-1,M-1])
a1, b1, c1 = calculate_coeff(1)
am_1, bm_1, cm_1 = calculate_coeff(M - 1)
matrx[0,0] = b1
matrx[0,1] = c1
matrx[M-2, M-3] = am_1
matrx[M-2, M-2] = bm_1
for i in range(2, M-1):
    a, b, c = calculate_coeff(i)
    matrx[i-1, i-2] = a
    matrx[i-1, i-1] = b
    matrx[i-1, i] = c
inverse_m = (np.matrix(matrx)).I

for i in range(M, 0, -1):
    # 迭代
    Fi = f_matrx[i, 1:M]
    Fi_1 = inverse_m * Fi.reshape((M-1, 1))
    Fi_1 = list(np.array(Fi_1.reshape(1, M-1))[0])
    f_matrx[i-1, 1:M]=Fi_1

i = np.round(S/delta_S, 0)
f_matrx[0, int(i)]