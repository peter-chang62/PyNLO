# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = []


# %% Imports


#%%

def n2_to_chi3():
    pass

def chi3_to_n2():
    pass

def calculate_nonlinear_susceptibility(self):
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

# %% Old
# %% Chi3 =====================================================================

# #--- Third Order Nonlinearity
# chi3 = 5200e-24

# #gamma3_pump_frq = (3/8)*(chi3 * 2*pi*pump_frq)/(c * n_eff(v=pump_frq) * A_eff(v=pump_frq))
# #g3_0 = (3/8)*(chi3/n_eff(z=0, v=v_grid))*(2*pi*v_grid/c)*(2/(e0*c*A_eff(z=0, v=v_grid)))
# g3_0 = (2*pi*v_grid*e0*chi3*A_eff(z=0, v=v_grid))/(e0*n_eff(z=0, v=v_grid)*c*A_eff(z=0, v=v_grid))**2
# def gamma3(z=0):
#     return g3_0
# gamma3 = None