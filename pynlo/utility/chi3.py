# -*- coding: utf-8 -*-
"""
Conversion functions and other calculators relevant to the 3rd order nonlinear
susceptibility.

"""

__all__ = ["gamma_to_g3", "g3_to_gamma", "g3_spm", "raman_t"]


# %% Imports

import numpy as np
from scipy.constants import pi, c, epsilon_0 as e0


#%% Converters

#---- Effective Nonlinearities g3 and gamma
def gamma_to_g3(v_grid, gamma):
    """
    Convert from the gamma to g3 nonlinear parameter.

    If the given gamma is frequency independent, the returned g3 parameter
    will cancel out the self-steeping effect.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    gamma : array_like
        The effective nonlinear parameter.

    Returns
    -------
    g3

    """
    return gamma / (3/2 * 2*pi*v_grid)

def g3_to_gamma(v_grid, g3):
    """
    Convert from the g3 to gamma nonlinear parameter.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    g3 : array_like
        The effective nonlinear parameter.

    Returns
    -------
    gamma

    """
    return g3 * (3/2 * 2*pi*v_grid)

#---- 3rd Order Nonlinear Susceptibility chi3 and Kerr Parameter
def n2_to_chi3(n, n2):
    return n2 * (4/3 * e0*c*n**2)

def chi3_to_n2(n, chi3):
    return chi3 / (4/3 * e0*c*n**2)

#---- Effective Nonlinearity gamma and Kerr Parameter
def n2_to_gamma(v_grid, a_eff, n2):
    return n2 * (2*pi*v_grid/(c*a_eff))

def gamma_to_n2(v_grid, a_eff, gamma):
    return gamma / (2*pi*v_grid/(c*a_eff))


# %% Nonlinearity

def g3_spm(n_eff, a_eff, chi3_eff):
    return 1/2 * chi3_eff/(e0*c**2 * n_eff**2 * a_eff)

def g3_split(n_eff, a_eff, chi3_eff):
    return np.array([1/2 * ((e0*a_eff)/(c*n_eff))**0.5 * chi3_eff,
                     1/(e0*c*n_eff*a_eff)**0.5]).T

def raman_t(t_grid, dt, weights, b_weights=None):
    """
    Calculate the time-domain Raman and instantaneous normalized nonlinear
    response function.

    Parameters
    ----------
    t_grid : array_like of float
        The time grid over which to calculate the nonlinear response function.
    dt : float
        The time grid step size.
    weights : array_like of float
        The contributions due to vibrational resonances in the material. Must
        be given as `[fraction, tau_1, tau_2]`, where `fraction` is the
        fractional contribution of the resonance to the total nonlinear
        response function, `tau_1` is the period of the vibrational frequency,
        and `tau_2` is the resonance's characteristic decay time. Enter more
        than one resonance using an (n, 3) shaped input array.
    b_weights : array_like of float, optional
        The contributions due to boson peaks found in amorphous materials.
        Must be given as `[fraction, tau_b]`, where `fraction` is the
        fractional contribution of the boson peak to the total nonlinear
        response function, and `tau_b` is the boson peak's characteristic
        decay time. Enter more than one peak using an (n, 2) shaped input
        array. The default behavior is to ignore this term.

    Returns
    -------
    nonlinear_t : ndarray of float
        The time-domain nonlinear response function.

    Notes
    -----
    These are the analytical formulations as summarized in section 2.3.3
    Agrawal's Nonlinear Fiber Optics [1]_. More accurate simulations may be
    obtainable using digitized spectral measurements, such as from [2]_.

    References
    ----------
    .. [1] Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
        CA;: Academic Press; 2019.

        https://doi.org/10.1016/B978-0-12-817042-7.00009-9

    .. [2] R.H. Stolen in "Raman Amplifiers for Telecommunications". See
        figure 2.10 and comments in reference 28.

        https://doi.org/10.1007/978-0-387-21583-9_2

    """
    t_grid = np.asarray(t_grid, dtype=float)
    n = t_grid.size
    assert t_grid[n//2] == 0, "The origin must be at the center of the time grid."

    #---- Raman Response
    def h_r(fraction, tau_1, tau_2):
        """
        The contributions due to vibrational resonances.

        Parameters
        ----------
        fraction : float
        tau_1 : float
        tau_2 : float

        Returns
        -------
        ndarray of float

        """
        h_r = tau_1*(tau_1**-2 + tau_2**-2)*np.exp(-t_grid/tau_2)*np.sin(t_grid/tau_1)
        h_r[t_grid < 0] = 0
        h_r /= np.sum(h_r * dt)
        h_r *= fraction
        return h_r

    def h_b(fraction, tau_b):
        """
        The contributions due to boson peaks.

        Parameters
        ----------
        fraction : float
        tau_b : float

        Returns
        -------
        ndarray of float

        """
        h_b = (2*tau_b - t_grid)/tau_b**2 * np.exp(-t_grid/tau_b)
        h_b[t_grid < 0] = 0
        h_b /= np.sum(h_b * dt)
        h_b *= fraction
        return h_b

    # Vibrational Modes
    raman_t = np.zeros_like(t_grid)
    weights = np.asarray(weights, dtype=float)
    if len(weights.shape) == 1:
        weights = weights[np.newaxis]
    for weight in weights:
        raman_t += h_r(*weight)

    # Boson Peaks
    if b_weights is not None:
        b_weights = np.asarray(b_weights, dtype=float)
        if len(b_weights.shape) == 1:
            b_weights = b_weights[np.newaxis]
        for b_weight in b_weights:
            raman_t += h_b(*b_weight)

    #---- Instantaneous Response
    delta_t = np.zeros_like(t_grid)
    delta_t[n//2] = 1
    delta_t /= np.sum(delta_t * dt)
    delta_t *= 1 - np.sum(raman_t * dt)

    nonlinear_t = raman_t + delta_t
    return nonlinear_t


# %% Solitons

def nonlinear_length():
    pass #TODO

def soliton_number():
    # copy over linear_length, and test against in chi1
    # (chi1.linear_length = soliton_number**2 * nonlinear_length)
    pass #TODO:

def soltion_period():
    pass #TODO

def soliton_fission():
    pass #TODO: estimate of soliton fission point, TD compression, and FD expansion

def soliton_DW_dk(v_grid, beta, v0=None):
    pass #TODO

def soliton_SFS():
    pass #TODO: cleo 2021 JTu3A.66, Analytical Expression of Raman Induced Soliton Self Frequency Shift