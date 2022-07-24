# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from monte_carlo.monte_carlo_class1 import MonteCarloOptionPricing
from datetime import datetime

t = datetime.timestamp(datetime.strptime('20220603-17:06:00',"%Y%m%d-%H:%M:%S"))
T = datetime.timestamp(datetime.strptime('20220604-20:30:00',"%Y%m%d-%H:%M:%S"))

T = (T-t)/60/60/24/365
# initialize parameters
S0 = 31000.0  # e.g. spot price = 35
K = 30467.32  # e.g. exercise price = 40
T = T  # e.g. one year
r = 0.08  # e.g. risk free rate = 1%
sigma = 0.7  # e.g. volatility = 5%
div_yield = 0.0  # e.g. dividend yield = 1%
no_of_slice = 91  # no. of slices PER YEAR e.g. quarterly adjusted or 252 trading days adjusted

barrier_price = 80.0  # barrier level for barrier options
parisian_barrier_days = 21  # no.of consecutive trading days required for parisian options

# optional parameter
simulation_rounds = int(10000)  # For monte carlo simulation, a large number of simulations required

# initialize instance
MC = MonteCarloOptionPricing(S0=S0,
                             K=K,
                             T=T,
                             r=r,
                             sigma=sigma,
                             div_yield=div_yield,
                             simulation_rounds=simulation_rounds,
                             no_of_slices=no_of_slice,
                             # fix_random_seed=True,
                             fix_random_seed=500)

# stochastic interest rate
# MC.vasicek_model(a=0.2, b=0.1, sigma_r=0.01)  # use Vasicek model
MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # use Cox Ingersoll Ross (CIR) model

# stochastic volatility (sigma)
MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)  # heston model

MC.stock_price_simulation()
# MT.stock_price_simulation_with_poisson_jump(jump_alpha=0.1, jump_std=0.25, poisson_lambda=0)
MC.european_call()
# MC.asian_avg_price_option(avg_method='arithmetic', option_type="call")
# MC.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
# MC.barrier_option(option_type="call",
#                   barrier_price=barrier_price,
#                   barrier_type="knock-in",
#                   barrier_direction="down",
#                   parisian_barrier_days=parisian_barrier_days)
#
# MC.look_back_european()

import sympy

x, y = sympy.symbols("x y")  # 申明未知数"x"和"y"

a = sympy.solve([3 * x - 2 * y - 3, x + 2 * y - 5], [x, y])  # 写入需要解的方程组

