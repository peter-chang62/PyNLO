# -*- coding: utf-8 -*-
"""
Module containing conversion functions and other calculators relevant to the
3rd order nonlinear susceptibility.

"""

__all__ = []


# %% Imports

import numpy as np
from scipy.constants import pi, c, epsilon_0 as e0


#%% Converters

#---- Effective Nonlinearities g3 and gamma
def gamma_to_g3(v_grid, gamma):
    return gamma / (3/2 * 2*pi*v_grid)

def g3_to_gamma(v_grid, g3):
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


# %% Estimators

def g3_spm(n_eff, a_eff, chi3_eff):
    return 1/2 * chi3_eff/(e0*c**2 * n_eff**2 * a_eff)

def g3_split(n_eff, a_eff, chi3_eff):
    return np.array([1/2 * ((e0*a_eff)/(c*n_eff))**0.5 * chi3_eff,
                     1/(e0*c*n_eff*a_eff)**0.5]).T

def calculate_raman(rt_grid, weights): #TODO
    '''Returns the RFFT of the normalized nonlinear susceptibilities chi2
    and chi3 in the frequency domain.

    The returned values are normalized such that sum(R*dt) = 1. Common
    functions are the instantaneous (dirac delta) and raman responses.

    Notes
    -----
    These relations only contain the nonlinear dispersion of the bulk
    material responses. Nonlinear dispersion attributable to the waveguide
    mode should be included in the g2 and g3 parameters.
    '''
    def dirac_delta():
        dd_t = np.zeros_like(self.nl_t_grid)
        dd_t[0] = 1/self.nl_dt
        dd_t = fftshift(dd_t)
        return dd_t

    def R_a(tau1, tau2, fraction):
        t_delay = self.nl_t_grid
        RT = ((tau1**2+tau2**2)/(tau1*tau2**2))*exp(t_delay/tau2)*np.sin(t_delay/tau1)
        RT[t_delay > 0] = 0
        RT *= fraction
        return RT

    #--- chi2
    # Instantaneous
    R2_v = mkl_fft.rfft_numpy(ifftshift(dirac_delta()) * self.nl_dt)

    #--- chi3
    # Instantaneous
    r3_v = mkl_fft.rfft_numpy(ifftshift(dirac_delta()) * self.nl_dt)

    # PPLN Raman Response
    # see arXiv:1211.1721 for coefficients
    raman_fraction = 0.2
    #           tau1        tau2        fraction
    weights = [[21e-15,     544e-15,    0.635],
               [19.3e-15,   1021e-15,   0.105],
               [15.9e-15,   1361e-15,   0.020],
               [8.3e-15,    544e-15,    0.240]]

    raman_t = np.zeros_like(self.nl_t_grid)
    for weight in weights:
        raman_t += R_a(*weight)

    raman_t *= 1/np.sum(raman_t * self.nl_dt)
    raman_v = mkl_fft.rfft_numpy(ifftshift(raman_t) * self.nl_dt)

    r3_v = (1.-raman_fraction)*r3_v + raman_fraction*raman_v

    return R2_v, r3_v


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
