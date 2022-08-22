import numpy as np
from py_vollib_vectorized import vectorized_black_scholes_merton

def local_vol(df):
    dt = 1 / 252
    dk = 1.5
    local_vol = np.zeros([1, len(df)])

    dc_by_dt = (vectorized_black_scholes_merton('c', df.s, df.k, df.t + dt, 0, df.iv, 0,
                                                return_as='array') - vectorized_black_scholes_merton('c', df.s, df.k,
                                                                                                     df.t, 0, df.iv, 0,
                                                                                                     return_as='array')) / dt

    dc_by_dk = (vectorized_black_scholes_merton('c', df.s, df.k + dk, df.t, 0, df.iv, 0,
                                                return_as='array') - vectorized_black_scholes_merton('c', df.s, df.k,
                                                                                                     df.t, 0, df.iv, 0,
                                                                                                     return_as='array')) / dk

    d2c_by_dk2 = (vectorized_black_scholes_merton('c', df.s, df.k + dk, df.t, 0, df.iv, 0,
                                                  return_as='array') - 2 * vectorized_black_scholes_merton('c', df.s,
                                                                                                           df.k, df.t,
                                                                                                           0, df.iv, 0,
                                                                                                           return_as='array') + vectorized_black_scholes_merton(
        'c', df.s, df.k - dk, df.t, 0, df.iv, 0, return_as='array')) / (dk ** 2)

    for i in range(len(df)):
        local_vol[0, i] = np.sqrt((dc_by_dt[i] + df.loc[i, 'r'] * df.loc[i, 'c'] - (df.loc[i, 'r'] - df.loc[i, 'q']) * (
                    df.loc[i, 'c'] - df.loc[i, 'k'] * dc_by_dk[i])) / (0.5 * df.loc[i, 'k'] ** 2 * d2c_by_dk2[i]))

    return local_vol