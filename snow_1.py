import numpy as np
from scipy import interpolate
import scipy as sp
import time


# -----------------------------------------------
# Class for all parameters
# -----------------------------------------------

class parameters():
    """parameters to used for pricing snowball option using monte carlo"""

    def __init__(self):
        """initialize parameters"""

        self.S = 6500  # underlying spot
        self.K = 6500  # strike
        self.KI_Barrier = 0.75  # down in barrier of put
        self.KO_Barrier = 1.03  # autocall barrier
        self.KO_Coupon = 0.25  # autocall coupon (p.a.)
        self.Bonus_Coupon = 0.25  # bonus coupon (p.a.)
        self.r = 0.03  # risk-free interest rate
        self.div = 0.01  # dividend rate
        self.repo = 0.08  # repo rate
        self.T = 1  # time to maturity in years
        self.v = 0.12  # volatility
        self.N = 252 * self.T  # number of discrete time points for whole tenor
        self.n = int(self.N / (self.T * 12))  # number of dicrete time point for each month
        self.M = int(self.T * 12)  # number of months
        self.dt = self.T / self.N  # delta t
        self.simulations = 30000
        self.J = 900  # number of steps of uly in the scheme
        self.lb = 0  # lower bound of domain of uly
        self.ub = 1.5  # upper bound of domain of uly
        self.dS = (self.ub - self.lb) / self.J  # delta S

    def print_parameters(self):
        """print parameters"""

        print("---------------------------------------------")
        print("Pricing a Snowball option using PDE")
        print("---------------------------------------------")
        print("Parameters of Snowball Option Pricer:")
        print("---------------------------------------------")
        print("Underlying Asset Price = ", self.S)
        print("Strike = ", self.K)
        print("Knock-in Barrier = ", self.KI_Barrier)
        print("Autocall Barrier = ", self.KO_Barrier)
        print("Autocall Coupon = ", self.KO_Coupon)
        print("Bonus Coupon = ", self.Bonus_Coupon)
        print("Risk-Free Rate =", self.r)
        print("Dividend Rate =", self.div)
        print("Repo Rate =", self.repo)
        print("Years Until Expiration = ", self.T)
        print("Volatility = ", self.v)
        print("Discrete time points =", self.N)
        print("Time-Step = ", self.dt)
        print("Underlyign domain = [", self.lb, ",", self.ub, "]")
        print("Discrete underlying points =", self.J)
        print("Underlying-Step = ", self.dS)
        print("---------------------------------------------")


class snowball_mc(parameters):

    def __init__(self):
        parameters.__init__(self)
        self.price_trajectories = []  # prices of all simulation paths
        self.option_price = np.nan  # option price using MC
        self.delta = np.nan
        self.gamma = np.nan
        self.vega = np.nan

    def compute_price(self):

        # reset trajectory
        self.price_trajectories = []

        # start simulation
        for i in range(self.simulations):
            e = sp.random.normal(0, 1, self.N)

            stockprices = np.cumprod(np.exp((self.r - self.div - self.repo -
                                             0.5 * self.v ** 2) * self.dt \
                                            + self.v * np.sqrt(self.dt) * e)) * self.S

            # check if Knockout, monthly observation
            n = int(1.0 / self.dt / 12.0)  # number of time points in every month
            s = slice(n - 1, self.N, n)
            stockprices_slice = stockprices[s]
            if stockprices_slice.max() >= self.KO_Barrier:
                idx = np.argmax(stockprices_slice >= self.KO_Barrier)
                time_to_KO = (idx + 1) / 12.0
                pv = (self.KO_Coupon * time_to_KO + 1) * np.exp(-self.r * time_to_KO)
                self.price_trajectories.append(pv)
                continue

            # if no KO, bonus coupon or down in put
            indicator_KI = stockprices.min() <= self.KI_Barrier
            pv = ((1 - indicator_KI) * (self.Bonus_Coupon * self.T + 1) \
                  + indicator_KI * (stockprices[-1] - self.K + 1)) * np.exp(-self.r * self.T)
            self.price_trajectories.append(pv)

        self.option_price = np.sum(self.price_trajectories) / self.simulations

    def compute_greeks(self):
        """"compute greeks of snowball option"""

        epsilon = 0.01
        S0 = self.S

        # price with S = S0 * (1 - epsilon)
        self.S = S0 * (1 - epsilon)
        self.compute_price()
        P1 = self.option_price

        # price with S = S0 * (1 + epsilon)
        self.S = S0 * (1 + epsilon)
        self.compute_price()
        P2 = self.option_price

        # price with S = S0 and vol = vol + epsilon
        self.S = S0
        self.v = self.v + epsilon
        self.compute_price()
        P3 = self.option_price

        # back to original and price option price
        self.v = self.v - epsilon
        self.compute_price()
        P0 = self.option_price

        self.delta = (P2 - P1) / (2 * S0 * epsilon)
        self.gamma = (P1 + P2 - 2 * P0) / (S0 ** 2 * epsilon ** 2)
        self.vega = (P3 - P0) / epsilon

class snowball_pde(parameters):

    def __init__(self):
        parameters.__init__(self)
        self.Mat_Inv = self.__getInvMat()  # inverse matrix used in backwardation
        self.option_price = np.nan
        self.__V = np.zeros((self.J + 1, self.N + 1))  # backwardation grid
        self.delta = np.nan
        self.gamma = np.nan
        self.vega = np.nan

    """" 3 helper function to calculate inverse matrix needed"""

    def __a0(self, x):
        return 0.5 * self.dt * ((self.r - self.div - self.repo) * x - self.v ** 2 * x ** 2)

    def __a1(self, x):
        return 1 + self.r * self.dt + self.v ** 2 * x ** 2 * self.dt

    def __a2(self, x):
        return 0.5 * self.dt * (-(self.r - self.div - self.repo) * x - self.v ** 2 * x ** 2)

    def __getInvMat(self):
        """Calculating Inverse Matrix"""

        # first line
        V = np.zeros((self.J + 1, self.J + 1))
        V[0, 0] = 1.0 / (1 - self.r * self.dt)

        # lines between
        for i in range(1, self.J):
            V[i, i - 1] = self.__a0(i)
            V[i, i] = self.__a1(i)
            V[i, i + 1] = self.__a2(i)

        # last line
        V[self.J, self.J - 1] = self.__a0(self.J) - self.__a2(self.J)
        V[self.J, self.J] = self.__a1(self.J) + 2 * self.__a2(self.J)

        return np.matrix(V).I

    def __interpolate_price(self, y, s):

        x = [self.lb + self.dS * i for i in range(self.J + 1)]
        f = interpolate.interp1d(x, y, kind='cubic')

        return float(f(s))

    def __compute_autocall(self):
        """present value of autocall coupon if KO"""

        # initialize payoff at maturity
        V_terminal = np.zeros((self.J + 1, 1))
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS), \
                         self.J + 1, 1)] = self.KO_Coupon + 1
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

            # pay coupon if KO at the end of each month
            KO_Coupon_temp = self.KO_Coupon * (self.T * 12 - i - 1) / 12
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS), self.J + 1, 1)] = KO_Coupon_temp + 1

        self.__V = self.__V + V_matrix

    def __compute_bonus(self):
        """present value of bonus coupon if not KO and not KI"""

        # initialize payoff at maturity
        V_terminal = np.zeros((self.J + 1, 1))
        V_terminal[slice(int((self.KI_Barrier - self.lb) / self.dS), \
                         int((self.KO_Barrier - self.lb) / self.dS), 1)] = self.Bonus_Coupon + 1
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

                # no bonus coupon if knock in, observed daily
                V_matrix[:, self.N - idx - 1][slice(0, \
                                                    int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0

            # no bonus coupon if knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS), self.J + 1, 1)] = 0

        self.__V = self.__V + V_matrix

    def __compute_put_UO(self):
        """value of put up and out"""

        # initialize payoff at maturity
        V_terminal = np.array([-1 + max(self.K - i * self.dS, 0) for i in range(self.J + 1)]).reshape((self.J + 1, 1))
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS) + 0, self.J + 1, 1)] = 0
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

            # nothing if Knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS) + 0, self.J + 1, 1)] = 0

        self.__V = self.__V - V_matrix

    def __compute_put_UO_DO(self):
        """value of put down&out and up&out"""

        # initialize payoff at maturity
        V_terminal = np.array([-1 + max(self.K - i * self.dS, 0) for i in range(self.J + 1)]).reshape((self.J + 1, 1))
        V_terminal[slice(0, int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0
        V_terminal[slice(int((self.KO_Barrier - self.lb) / self.dS) + 1, self.J + 1, 1)] = 0
        V_matrix = np.zeros((self.J + 1, self.N + 1))
        V_matrix[:, -1] = V_terminal.reshape((self.J + 1,))

        # backwardation
        for i in range(self.M):
            for j in range(self.n):
                idx = i * self.n + j
                V_matrix[:, self.N - idx - 1] = (self.Mat_Inv * \
                                                 V_matrix[:, self.N - idx].reshape((self.J + 1, 1))).reshape(
                    (self.J + 1,))

                # nothing if knock in, observed daily
                V_matrix[:, self.N - idx - 1] \
                    [slice(0, int((self.KI_Barrier - self.lb) / self.dS), 1)] = 0

            # nothing if knock out, observed monthly
            if i != self.M - 1:
                V_matrix[:, self.N - idx - 1] \
                    [slice(int((self.KO_Barrier - self.lb) / self.dS) + 1, self.J + 1, 1)] = 0

        self.__V = self.__V + V_matrix

    def compute_price(self):
        """compute the price of snowball option"""

        # reset inverse Matrix in case vol has been changed
        self.Mat_Inv = self.__getInvMat()

        self.__V = np.zeros((self.J + 1, self.N + 1))
        self.__compute_autocall()
        self.__compute_bonus()
        self.__compute_put_UO()
        self.__compute_put_UO_DO()

        self.option_price = self.__interpolate_price(self.__V[:, 0], self.S)

    def compute_greeks(self):
        """"compute greeks of snowball option"""

        epsilon = 0.01
        self.v = self.v + epsilon
        self.compute_price()
        P3 = self.option_price

        # back to original and price
        self.v = self.v - epsilon
        self.compute_price()
        P0 = self.option_price
        P1 = self.__interpolate_price(self.__V[:, 0], self.S * (1 - epsilon))
        P2 = self.__interpolate_price(self.__V[:, 0], self.S * (1 + epsilon))

        self.delta = (P2 - P1) / (2 * self.S * epsilon)
        self.gamma = (P1 + P2 - 2 * P0) / (self.S ** 2 * epsilon ** 2)
        self.vega = (P3 - P0) / epsilon


# -----------------------------------------------
# Testing
# -----------------------------------------------
if __name__ == '__main__':
    pde = snowball_pde()
    pde.print_parameters()
    tic = time.time()
    print("Starting calculating......", end="")
    pde.compute_price()
    print("Done.")
    print("Option price = ", pde.option_price)
    print("Running time = ", time.time() - tic, "s")
    print("---------------------------------------------")

    tic = time.time()
    print("Calculating Greeks.....", end="")
    pde.compute_greeks()
    print("Done.")
    print("Option delta = ", pde.delta)
    print("Option gamma = ", pde.gamma)
    print("Option vega = ", pde.vega)
    print("Running time = ", time.time() - tic, "s")
    print("---------------------------------------------")