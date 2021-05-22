# -*- coding: utf-8 -*-
"""
This example demonstrates supercontinuum generation due to highly
phase-mismatched second harmonic generation in periodically polled lithium
niobate.

"""

# %% Imports

import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

import pynlo
from pynlo import utility as utils


# %% Pulse Properties

n_points = 2**13
v_min = c/3500e-9   # c / 3500 nm
v_max = c/450e-9    # c / 900 nm
v0 = c/1550e-9      # c / 1550 nm
e_p = 1e-9          # 1 nJ
t_fwhm = 50e-15     # 50 fs

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
pulse.rtf_grids(n_harmonic=2, update=True) # anti-aliasing

v_grid = pulse.v_grid
t_grid = pulse.t_grid


# %% Waveguide Properties

length = 10e-3 # 10 mm
a_eff = 10e-6 * 10e-6 # 10 um * 10 um

#---- Wavenumbers
def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis=="e":
        a1 = 5.756 # plasmons in the far UV
        a2 = 0.0983 # weight of UV pole
        a3 = 0.2020 # pole in UV
        a4 = 189.32 # weight of IR pole
        a5 = 12.52 # pole in IR
        a6 = 1.32e-2 # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis=="o":
        a1 = 5.653 # plasmons in the far UV
        a2 = 0.1185 # weight of UV pole
        a3 = 0.2091 # pole in UV
        a4 = 89.61 # weight of IR pole
        a5 = 10.85 # pole in IR
        a6 = 1.97e-2 # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    wvl = c/v * 1e6 # um
    f = (T-24.5)*(T+570.82)
    n2 = ((a1 + b1*f) + (a2 + b2*f)/(wvl**2 - (a3 + b3*f)**2)
          + (a4 + b4*f)/(wvl**2 - a5**2) - a6*wvl**2)
    return n2**0.5
n_eff = n_MgLN_G(v_grid)

beta = n_eff * 2*pi*v_grid/c

#---- 2nd order nonlinearity
d_eff = 27e-12 # 27 pm / V
chi2_eff = 2 * d_eff
g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)

# Polling
p_0 = 31e-6 # 31 um
def n_cycles(z):
    """integral(dz/period(z))"""
    return z/p_0
def z_inversion(n):
    """the z position of the nth polling inversion"""
    return 0.5*n*p_0
z_invs = z_inversion(np.arange(int(n_cycles(length)*2)+1)) # all inversion points
z_grid = np.append(z_invs[z_invs < length],length) # include last point

g2_poll = utils.chi2.polling_sign(n_cycles)

#---- 3rd order nonlinearity
chi3_eff = 5200e-24 # 5200 pm**2 / V**2
g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)

mode = pynlo.media.Mode(v_grid, beta, g2_v=g2, g2_inv=g2_poll, g3_v=g3, z=0.0)


# %% Model

model = pynlo.model.SM_UPE(pulse, mode)

#---- Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(n=20, local_error=local_error)


# %% Simulate

pulse_out, z, a_t, a_v = model.simulate(
    z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)


# %% Plot Results

fig = plt.figure("Simulation Results", clear=True)
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

p_v_dB = 10*np.log10(np.abs(a_v)**2)
p_v_dB -= p_v_dB.max()
ax0.plot(1e-12*v_grid, p_v_dB[0], color="b")
ax0.plot(1e-12*v_grid, p_v_dB[-1], color="g")
ax2.pcolormesh(1e-12*v_grid, 1e3*z, p_v_dB, vmin=-40.0, vmax=0, shading="auto")
ax0.set_ylim(bottom=-50, top=10)
ax2.set_xlabel('Frequency (THz)')

p_t_dB = 10*np.log10(np.abs(a_t)**2)
p_t_dB -= p_t_dB.max()
ax1.plot(1e12*t_grid, p_t_dB[0], color="b")
ax1.plot(1e12*t_grid, p_t_dB[-1], color="g")
ax3.pcolormesh(1e12*t_grid, 1e3*z, p_t_dB, vmin=-40.0, vmax=0, shading="auto")
ax1.set_ylim(bottom=-50, top=10)
ax3.set_xlabel('Time (ps)')

ax0.set_ylabel('Power (dB)')
ax2.set_ylabel('Propagation Distance (mm)')
fig.tight_layout()
fig.show()
