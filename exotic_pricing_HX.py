# -*- coding: utf-8 -*-
# @Time    : 2022/08/12 17:55
# @Author  : Timo Yang
# @Email   : timo_yang@digifinex.org
# @File    : exotic_pricing_HX.py
# @Software: PyCharm

import numpy as np
from datetime import datetime
from typing import Tuple

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
        # self.sigma = float(sigma)

        # MC params
        self.npath = int(npath)
        self.simulation_rounds = int(simulation_rounds)

        self._dt = self.T / self.simulation_rounds # the time interval in simulation

        self.mue = r  # under risk-neutral measure, asset expected return = risk-free rate

        self.terminal_prices = [] # simulation price at T

        self.z_t = np.random.standard_normal((self.simulation_rounds, self.npath))

        self.sigma = np.full((self.simulation_rounds, self.npath), sigma)

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

    def heston(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0) -> np.ndarray:
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
        _CH = np.linalg.cholesky(_cov) # Cholesky decomposition

        for i in range(1, self.simulation_rounds):
            _ZH = np.random.normal(size=(2, self.npath//2))
            _ZA = np.c_[_ZH,-_ZH] # Antithetic sampling
            _Z = _CH @ _ZA
            _dS = self.r * self.S0 * self._dt + np.sqrt(_V[i-1, :]) * _S[i-1, :] * np.sqrt(self._dt) * _Z[0, :]
            _S[i ,:] = _S[i-1,:] + _dS
            _dV = kappa * (theta - _V[i-1, :]) * self._dt + sigma_v * np.sqrt(_V[i-1, :]) * np.sqrt(self._dt) * _Z[1, :]
            _V[i, :] = np.maximum(_V[i-1, :] + _dV, 0)

        self.price_array = _S
        self.terminal_prices = _S[-1,:]
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))
        self.sigma = _V
        print('MC price expection:', np.mean(self.terminal_prices),'\nMC price sigma:', self.stock_price_standard_error)
        return np.mean(self.terminal_prices), self.stock_price_standard_error

    def stock_price_simulation(self) -> Tuple[np.ndarray, float]:
        self.exp_mean = (self.mue - (self.sigma ** 2.0) * 0.5) * self._dt
        self.exp_diffusion = self.sigma * np.sqrt(self._dt)

        self.price_array = np.zeros((self.simulation_rounds,self.npath))
        self.price_array[0, :] = self.S0

        for i in range(1, self.simulation_rounds):
            self.price_array[i, :] = self.price_array[i - 1, :] * np.exp(
                self.exp_mean[i,:] + self.exp_diffusion[i - 1, :] * self.z_t[i - 1, :]
            )

        self.terminal_prices = self.price_array[-1, :]
        self.stock_price_expectation = np.mean(self.terminal_prices)
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))

        print('-' * 64)
        print(
            " Number of simulations %4.1i \n S0 %4.1f \n K %2.1f \n Maximum Stock price %4.2f \n"
            " Minimum Stock price %4.2f \n Average stock price %4.3f \n Standard Error %4.5f " % (
                self.simulation_rounds, self.S0, self.K, np.max(self.terminal_prices),
                np.min(self.terminal_prices), self.stock_price_expectation, self.stock_price_standard_error
            )
        )
        print('-' * 64)

        return self.stock_price_expectation, self.stock_price_standard_error

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
                for i in range(0, MC.simulation_rounds - days_to_slices):
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

    def binary_option(self, option_type, payoff,loss):
        if option_type == 'call':
            self.terminal_profit = np.where(self.terminal_prices > self.K, payoff, loss)
            self.expectation = np.mean(self.terminal_profit * np.exp(-self.r * self.T))
            self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))
        else:
            self.terminal_profit = np.where(self.terminal_prices < self.K, payoff, loss)
            self.expectation = np.mean(self.terminal_profit * np.exp(-self.r * self.T))
            self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Binary european %s \n Payoff: %s \n Loss: %s \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, payoff, loss,
                self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)
        return self.expectation, self.standard_error

    def look_back_european(self, option_type: str = 'call') -> Tuple[float, float]:
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'

        self.max_price = np.max(self.price_array, axis=0)
        self.min_price = np.min(self.price_array, axis=0)

        if option_type == "call":
            self.terminal_profit = np.maximum((self.max_price - self.K), 0.0)
        elif option_type == "put":
            self.terminal_profit = np.maximum((self.K - self.min_price), 0.0)

        self.expectation = np.mean(self.terminal_profit * np.exp(-self.r * self.T))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Lookback european %s monte carlo \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, self.S0, self.K, self.expectation, self.standard_error
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
        print('-' * 64)
        print(
            " Snowball monte carlo \n S0 %4.1f \n K %2.1f \n Knock-in price %4.1f \n Knock-out price %4.1f \n"
            " Option Value %4.3f" % (
                self.S0, self.K, KI_Barrier * self.S0, KO_Barrier * self.S0, _option_price
            )
        )
        print('-' * 64)
        return _option_price

if __name__ == '__main__':
    # Test
    t = datetime.timestamp(datetime.strptime('20210603-00:00:00', "%Y%m%d-%H:%M:%S"))
    T = datetime.timestamp(datetime.strptime('20220603-00:00:00', "%Y%m%d-%H:%M:%S"))
    T = (T - t) / 60 / 60 / 24 / 365

    # initialize parameters
    S0 = 35  # e.g. spot price = 35
    K = 40  # e.g. exercise price = 40
    T = T  # e.g. one year
    r = 0.01  # e.g. risk free rate = 1%
    sigma = 0.5  # e.g. volatility = 5%
    npath = 10000  # no. of slices PER YEAR e.g. quarterly adjusted or 252 trading days adjusted

    # optional parameter
    simulation_rounds = int(100)  # For monte carlo simulation, a large number of simulations required

    MC = ExoticPricing(S0=S0,
                       K=K,
                       T=T,
                       r=r,
                       sigma=sigma,
                       simulation_rounds=simulation_rounds,
                       npath=npath,
                       fix_random_seed=501)

    MC.stock_price_simulation()
    MC.CIR_model(a=0.5, b=0.05, sigma_r=0.1)
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)

    # barrier
    # barrier_price= 80.0
    # parisian_barrier_days=21
    # MC.barrier_option(option_type="call",
    #                   barrier_price=barrier_price,
    #                   barrier_type="knock-in",
    #                   barrier_direction="down",
    #                   parisian_barrier_days=parisian_barrier_days)

    # binary
    payoff = 25
    loss = -10
    MC.binary_option(option_type='call',
                     payoff=payoff,
                     loss=loss)

    # look back
    MC.look_back_european('call')

    MC.snowball(KO_Barrier=1.3, KO_Coupon=0.25, KI_Barrier=0.75, Bonus_Coupon=0.25)