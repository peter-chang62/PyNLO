# -*- coding: utf-8 -*-
"""
Module containing conversion functions and other calculators relevant to the
linear susceptibility.

"""

__all__ = []


# %% Imports

import numpy as np
from scipy.constants import pi, c


#%% Converters

# TODO: forward and backward transformations, test with equivalents from media.Mode

#---- Propagation Constant and Linear Susceptibility chi1
def chi1_to_k(v_grid, chi1):
    k = 2*pi*v_grid/c * (1 + chi1)**0.5
    alpha = 2*k.imag
    beta = k.real
    return beta, alpha

def k_to_chi1(v_grid, beta, alpha=None):
    if alpha is None:
        k = beta
    else:
        k = beta + 1j/2 * alpha
    return (c/(2*pi*v_grid) * k)**2 - 1

#---- Wavenumber and Refractive Index
def n_to_beta(v_grid, n):
    return n * (2*pi*v_grid/c)

def beta_to_n(v_grid, beta):
    return beta / (2*pi*v_grid/c)

#---- GVD and Dispersion Parameter D
def D_to_beta2(v_grid, D):
    return D / (-2*pi * v_grid**2/c)

def beta2_to_D(v_grid, beta2):
    return beta2 * (-2*pi * v_grid**2/c)


# %% Helper Functions

def linear_length():
    pass #TODO