# -*- coding: utf-8 -*-
# @Time    : 2022/08/12 17:55
# @Author  : Timo Yang
# @Email   : timo_yang@digifinex.org
# @File    : exotic_pricing_HX.py
# @Software: PyCharm

import numpy as np
from datetime import datetime
from typing import Tuple
import matplotlib.pyplot as plt

class ExoticPricing:
    def __init__(self, r, S0: float, K: float, T: float, sigma: float,
                 simulation_rounds: int = 100, npath: int = 100000, fix_random_seed: bool or int = False):
        '''
        Parameters
        ----------
        :param S0: current price of the underlying asset (e.g. stock)
        :param K: exercise price
        :param T: time to maturity, in years, can be float
        :param r: interest rate, by default we assume constant interest rate model
        :param sigma: volatility (in standard deviation) of the asset annual returns
        :param simulation_rounds: the number of simulation
        :param npath: the lenth of MC simulation
        :param fix_random_seed: boolean or integer
        '''

        assert S0 >= 0, 'underlying price should be positive'
        assert T >= 0, 'time to marturity should be positive'
        assert sigma >= 0, 'vol of underlying assets should be positive'
        assert npath >= 0, 'no of slices per year should be positive'
        assert simulation_rounds >= 0, 'simulation rounds should be positive'

        # basic params
        self.r = float(r)
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.sigma = float(sigma)

        # MC params
        self.npath = int(npath)
        self.simulation_rounds = int(simulation_rounds)

        self._dt = self.T / self.simulation_rounds # the time interval in simulation

        self.mue = r  # under risk-neutral measure, asset expected return = risk-free rate

        self.terminal_prices = [] # simulation price at T

        # self.z_t = np.random.standard_normal((self.simulation_rounds, self.npath)) # dW in dt is a standard gaussian distribution

        if type(fix_random_seed) is bool:
            if fix_random_seed:
                np.random.seed(15000)
        elif type(fix_random_seed) is int:
            np.random.seed(fix_random_seed)

    def CIR_model(self, a: float, b: float, sigma_r: float) -> np.ndarray:
        '''
        CIR process assumes that the transition Prob Dist. of interest follows a non-central chi-square Dist.
        The degree of freedom for this chi-square Dist. is 4 * b * a / sigma_r ** 2

        Parameters
        ----------
        a: speed of mean-reversion
        b: long_term mean
        sigma_r: interest rate volatility (standard deviation)

        Returns
        -------

        '''
        assert 2 * a * b > sigma_r ** 2  # Feller condition, to ensure r_t > 0
        _interest_array = np.full((self.simulation_rounds, self.npath), self.r * self._dt)

        # CIR non-central chi-square distribution degree of freedom
        _dof = 4 * b * a / sigma_r ** 2

        for i in range(1, self.simulation_rounds):
            _Lambda = (4 * a * np.exp(-a * self._dt) * _interest_array[i - 1,:] / (
                    sigma_r ** 2 * (1 - np.exp(-a * self._dt))))
            _chi_square_factor = np.random.noncentral_chisquare(df=_dof,
                                                                nonc=_Lambda,
                                                                size=self.npath)

            _interest_array[i, :] = sigma_r ** 2 * (1 - np.exp(-a * self._dt)) / (
                    4 * a) * _chi_square_factor

        # re-define the interest rate array
        self.r = _interest_array
        return _interest_array

    def heston(self, plot: bool, kappa: float, theta: float, sigma_v: float, rho: float = 0.0) -> np.ndarray:
        '''
        Heston is a classic stochastic vol model, assuming the volatility follows Ornstein-Uhlenbeck process(mean-reversion),
        it describes the correlation between the vol of underlying asset and the implied vol of options.

        Parameters
        ----------
        :param: kappa: rate at which vt reverts to theta
        :param: theta: long-term variance
        :param: sigma_v: sigma of the volatility
        :param: rho: correlation between the volatility and the rate of return

        Returns
        -------
        dv(t) = kappa[theta - v(t)] * dt + sigma_v * sqrt(v(t)) * dZ
        '''

        _variance_v = sigma_v ** 2
        assert 2 * kappa * theta > _variance_v, 'Feller condition is not satisfied, check the parameters!'  # Feller condition

        _S = self.S0 * np.ones((self.simulation_rounds, self.npath))
        _V = self.sigma * np.ones((self.simulation_rounds, self.npath))
        _cov = np.array([[1, rho], [rho, 1]])
        _CH = np.linalg.cholesky(_cov)  # Cholesky decomposition

        for i in range(1, self.simulation_rounds):
            _ZH = np.random.normal(size=(2, self.npath // 2))
            _ZA = np.c_[_ZH, -_ZH]  # Antithetic sampling
            _Z = _CH @ _ZA
            _dS = self.r * self.S0 * self._dt + np.sqrt(_V[i - 1, :]) * _S[i - 1, :] * np.sqrt(self._dt) * _Z[0, :]
            _S[i, :] = _S[i - 1, :] + _dS
            _dV = kappa * (theta - _V[i - 1, :]) * self._dt + sigma_v * np.sqrt(_V[i - 1, :]) * np.sqrt(self._dt) * _Z[
                                                                                                                    1,
                                                                                                                    :]
            _V[i, :] = np.maximum(_V[i - 1, :] + _dV, 0)

        self.price_array = _S
        self.terminal_prices = _S[-1, :]
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))
        self.sigma = _V
        print('MC price expection:', np.mean(self.terminal_prices), '\nMC price sigma:',
              self.stock_price_standard_error)

        if plot == True:
            fig = plt.figure(figsize=(28, 16))
            plt.plot(np.arange(0, 1000), self.price_array)
            plt.show()
        else:
            pass

        return np.mean(self.terminal_prices), self.stock_price_standard_error

    def barrier_option(self, option_type: str, barrier_price: float, barrier_type: str, barrier_direction: str,
                       parisian_barrier_days: int or None = None) -> Tuple[float, float]:
        '''

        Parameters
        ----------
        option_type: 'call' or 'put'
        barrier_price: barrier price
        barrier_type: 'knock_in' or 'knock_out'
        barrier_direction: 'up' or 'down'
        parisian_barrier_days: a continuously period, parisian options can be excercised only if
                               the underlying price remain higher/lower than barrier_price in this 'period' before the excercise day

        Returns
        -------

        '''
        assert option_type == "call" or option_type == "put", 'option type must be either call or put'
        assert barrier_type == "knock-in" or barrier_type == "knock-out", \
            'barrier type must be either knock-in or knock-out'
        assert barrier_direction == "up" or barrier_direction == "down", \
            'barrier direction must be either up or down'

        if barrier_direction == "up":
            barrier_check = np.where(self.price_array >= barrier_price, 1, 0) # Indicator Matrix

            if parisian_barrier_days is not None:
                days_to_slices = int(parisian_barrier_days * self.simulation_rounds / (self.T * 365))
                parisian_barrier_check = np.zeros((self.simulation_rounds - days_to_slices, self.npath))
                for i in range(0, self.simulation_rounds - days_to_slices):
                    '''
                    模拟路径分段，找到是否有生成的价格成功超过/低于巴黎期权设定的时间段
                    '''
                    parisian_barrier_check[i, :] = np.where(np.sum(barrier_check[i:i + days_to_slices, :], axis=0) >= days_to_slices, 1, 0)

                barrier_check = parisian_barrier_check

        elif barrier_direction == "down":
            barrier_check = np.where(self.price_array <= barrier_price, 1, 0) # Indicator Matrix

            if parisian_barrier_days is not None:
                days_to_slices = int(parisian_barrier_days * self.simulation_rounds / (self.T * 365))
                parisian_barrier_check = np.zeros((self.simulation_rounds - days_to_slices, self.npath))
                for i in range(0, self.simulation_rounds - days_to_slices):
                    parisian_barrier_check[i, :] = np.where(np.sum(barrier_check[i:i + days_to_slices, :], axis=0) >= days_to_slices, 1, 0)

                barrier_check = parisian_barrier_check

        if option_type == 'call':
            self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        elif option_type == 'put':
            self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)

        if barrier_type == "knock-in":
            self.terminal_profit = np.where(barrier_check[-1:] >= 1, self.intrinsic_val[-1, :], 0)
        elif barrier_type == "knock-out":
            self.terminal_profit = np.where(barrier_check[-1:] >= 1, 0, self.intrinsic_val[-1, :])

        self.expectation = np.mean(self.terminal_profit * np.exp(-self.r * self.T))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit[0]))

        print('-' * 64)
        print(
            " Barrier european %s \n Type: %s \n Direction: %s @ %s \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, barrier_type, barrier_direction, barrier_price,
                self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def snowball(self, KO_Barrier, KO_Coupon, KI_Barrier, Bonus_Coupon):

        self.price_trajectories = []

        for i in range(self.npath):
            _n = int(1.0 / self._dt / 12.0)  # number of time points in every month
            _s = np.arange(int(_n * 3), self.simulation_rounds, int(self.simulation_rounds/12))
            _stockprices_slice = self.price_array[_s, i]
            if _stockprices_slice.max() >= KO_Barrier:
                idx = np.argmax(_stockprices_slice >= KO_Barrier)
                time_to_KO = (idx + 1) / 12.0
                pv = (KO_Coupon * time_to_KO + 1) * np.exp(-r * time_to_KO)
                self.price_trajectories.append(pv)
                continue

            # if no KO, bonus coupon or down in put
            _stockprices = self.price_array[:,i]
            indicator_KI = _stockprices.min() <= KI_Barrier
            pv = ((1 - indicator_KI) * (Bonus_Coupon * T + 1) + indicator_KI * (_stockprices[-1] - K + 1)) * np.exp(-r * T)
            self.price_trajectories.append(pv)

        _option_price = np.sum(self.price_trajectories) / self.npath
        return _option_price


# Test
t = datetime.timestamp(datetime.strptime('20210603-00:00:00',"%Y%m%d-%H:%M:%S"))
T = datetime.timestamp(datetime.strptime('20220603-00:00:00',"%Y%m%d-%H:%M:%S"))
T = (T-t)/60/60/24/365

# initialize parameters
S0 = 6500 # e.g. spot price = 35
K = 6500  # e.g. exercise price = 40
T = T  # e.g. one year
r = 0.01  # e.g. risk free rate = 1%
sigma = 0.5  # e.g. volatility = 5%
npath = 10000

# optional parameter
simulation_rounds = int(1000)

MC = ExoticPricing(S0=S0,
                   K=K,
                   T=T,
                   r=r,
                   sigma=sigma,
                   simulation_rounds=simulation_rounds,
                   npath=npath,
                   fix_random_seed=502)

MC.heston(plot = False, kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)

# 检查时间点
# 以每日检查敲入，三个月后开始检查敲出为例
knock_out_check = np.arange(int(3/12/MC._dt),
                            int(MC.simulation_rounds),
                            int(1/(MC._dt * 12)))

knock_in_check = np.arange(int(0),
                           int(MC.simulation_rounds),
                           int(1/(MC._dt * 365))) # simulation rounds至少要到1000才较为精确

# 敲出
# 1. 首先找出所有时点上，价格超过knock out点的路径
knock_out_indicator = np.where(MC.price_array > S0 * 1.3, 1, 0)
indicator = np.zeros([len(knock_out_check), MC.npath])

# 2. 筛选出在检查时间点上超过knock out点的路径
j=0
for i in knock_out_check:
    tmp = np.where((knock_out_indicator[i, :] == 1), 1, 0)
    indicator[j,:] = tmp
    j += 1

for i in range(1,indicator.shape[0]):
    for j in range(0,indicator.shape[1]):
        if indicator[0:i,j].any() == 1:
            indicator[i, j] = 0
        else:pass
num = np.sum(indicator,axis=1)

# 3. 计算总收益
# num = np.sum(num)
terminal_profit1 = np.zeros_like(num)
for i in range(len(num)):
    prof = (1+0.25) * 10000 * num[i] * (3+i)/12 * np.exp(-r * (3+i)/12)
    terminal_profit1[i]=prof
expection1 = terminal_profit1 @ (num/MC.npath).T

profit_array1 = []
for i in range(len(num)):
    prof = [0.25 * 10000 * (3+i)/12 * np.exp(-r * (3+i)/12)] * int(num[i])
    profit_array1.append(prof)

profit_array1=[i for j in profit_array1 for i in j ]
# 未发生敲入敲出
# 1. 首先找出所有时点上，价格小于knock out且大于knock in点的路径
between_indicator = np.where((MC.price_array < S0 * 1.3) & (S0 * 0.8 < MC.price_array), 1, 0)

# 2. 符合此条件的路径，收益都相同，为持有一年获得的利润
num2 = 0
for i in range(0,between_indicator.shape[1]):
    if between_indicator[250:,i].all() == 1:
        num2 += 1
    else:pass

terminal_profit2 = 10000 * (1+0.25) * num2
expection2 = terminal_profit2 * np.exp(-r * T) * num2/MC.npath
profit_array2 = [terminal_profit2] * num2

# 发生敲入，且到期价格落在初始和敲出价格区间之内,获利0元
indicator3 = np.zeros([1,MC.npath])
for i in range(MC.price_array.shape[1]):
    if ((MC.price_array[[knock_in_check], i]).any() < S0*0.8) & (S0 < MC.price_array[-1,i] < S0*1.3):
        indicator3[0,i] = int(1)
    else:pass

for i in range(indicator3.shape[1]):
    if tmp[i] == indicator3[0,i]:
        indicator3[0,i] = 0
    else:pass
num3 = indicator3.tolist()[0].count(1)
profit_array3 = [0] * num3

# 发生敲入，且到期价格低于期初价格,亏损(ST/S0 - 1)*本金
indicator4 = np.zeros([1,MC.npath])
for i in range(MC.price_array.shape[1]):
    if ((MC.price_array[[knock_in_check],i]).any() < S0*0.8) & (MC.price_array[-1,i] < S0):
        indicator4[0,i] = int(1)
    else:pass

tmp = np.sum(indicator,axis=0)
for i in range(indicator4.shape[1]):
    if tmp[i] == indicator4[0,i]:
        indicator4[0,i] = 0
    else:pass
num4 = indicator4.tolist()[0].count(1)

terminal_price = indicator4 * MC.price_array[-1,:]
terminal_price = terminal_price[terminal_price!=0]
terminal_profit4 = ((terminal_price-S0)/S0) * 10000
# terminal_profit4 = np.where(terminal_profit4 == -10000.00000, 0, terminal_profit4)
expection4 = np.sum(terminal_profit4 * np.exp(-r * T)) * num4/MC.npath
profit_array4 = terminal_profit4.copy()

profit_array = [profit_array1,profit_array2,profit_array3,profit_array4]
profit_array=[i for j in profit_array for i in j ]

np.sum(num) + num2 + num3 + num4

a = MC.snowball(1.03, 0.25, 0.75 , 0.25)