import numpy as np
import scipy.stats as sts

def heston(kappa: float, theta: float, sigma_v: float, sigma: float, T: float, simulation_rounds: int, no_of_slices: int, rho: float = 0.0) -> np.ndarray:
    """
    When asset volatility (variance NOT sigma!) follows a stochastic process.

    Heston stochastic volatility with Euler discretisation. Notice the native Euler discretisation could lead to
    negative volatility. To mitigate this issue, several methods could be used. Here we choose the full truncation
    method.

    dv(t) = kappa[theta - v(t)] * dt + sigma_v * sqrt(v(t)) * dZ
    :param: kappa: rate at which vt reverts to theta
    :param: theta: long-term variance(Expection of long-term var)
    :param: sigma_v: sigma of the volatility
    :param: rho: correlation between the volatility and the rate of return
    :return: stochastic volatility array
    """
    dt = T/no_of_slices
    sigma = np.full((simulation_rounds, no_of_slices), sigma)
    variance_v = sigma_v ** 2
    assert 2 * kappa * theta > variance_v  # Feller condition

    # step 1: simulate correlated zt
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])

    zt = sts.multivariate_normal.rvs(mean=mu, cov=cov, size=(simulation_rounds, no_of_slices))
    variance_array = np.full((simulation_rounds, no_of_slices),
                              sigma[0, 0] ** 2)

    z_t = zt[:, :, 0]
    zt_v = zt[:, :, 1]

    # step 2: simulation
    for i in range(1, no_of_slices):
        previous_slice_variance = np.maximum(variance_array[:, i - 1], 0)
        drift = kappa * (theta - previous_slice_variance) * dt
        diffusion = sigma_v * np.sqrt(previous_slice_variance) * \
                     zt_v[:, i - 1]
        delta_vt = drift + diffusion
        variance_array[:, i] = variance_array[:, i - 1] + delta_vt

    # re-define the interest rate and volatility path
    sigma = np.sqrt(np.maximum(variance_array, 0))

    return sigma

kappa, theta, sigma_v, rho=2,0.3,0.3,0.5
T=0.25
no_of_slices = 100
simulation_rounds = 10000
sigma = 0.7