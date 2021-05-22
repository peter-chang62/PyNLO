# -*- coding: utf-8 -*-
"""
This example demonstrates supercontinuum generation due to self-phase
modulation and soliton effects in a silica-based photonic crystal fiber.

"""

# %% Imports

import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

import pynlo
from pynlo import utility as utils


# %% Pulse Properties

n_points = 2**12
v_min = c/1400e-9   # c / 1400 nm
v_max = c/450e-9    # c / 450 nm
v0 = c/835e-9       # c / 835 nm
e_p = 550e-12       # 550 pJ
t_fwhm = 50e-15     # 50 fs

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
pulse.rtf_grids(n_harmonic=2, update=True) # anti-aliasing

v_grid = pulse.v_grid
t_grid = pulse.t_grid


# %% Waveguide Properties

length = 15e-2 # 15 cm

#---- Wavenumbers
beta_n = 11*[0]
beta_n[2] = -11.830 * 1e-12**2/1e3       # -11.830 ps**2 / km
beta_n[3] = 8.1038e-2 * 1e-12**3/1e3     # 8.1038e-2 ps**3 / km
beta_n[4] = -9.5205e-5 * 1e-12**4/1e3    # -9.5205e-5 ps**4 / km
beta_n[5] = 2.0737e-7 * 1e-12**5/1e3     # 2.0737e-7 ps**5 / km
beta_n[6] = -5.3943e-10 * 1e-12**6/1e3   # -5.3943e-10 ps**6 / km
beta_n[7] = 1.3486e-12 * 1e-12**7/1e3    # 1.3486e-12 ps**7 / km
beta_n[8] = -2.5495e-15 * 1e-12**8/1e3   # -2.5495e-15 ps**8 / km
beta_n[9] = 3.0524e-18 * 1e-12**9/1e3    # 3.0524e-18 ps**9 / km
beta_n[10] = -1.7140e-21 * 1e-12**10/1e3 # -1.7140e-21 ps**10 / km

beta = utils.taylor_series(2*pi*v0, beta_n)(2*pi*v_grid)

#---- 3rd order nonlinearity
gamma = 0.11        # 0.11 / W * m
t_shock = 0.56e-15  # 0.56 fs
g3 = utils.chi3.gamma_to_g3(v_grid, gamma, t_shock)

# Raman effect
rv_grid = pulse.rv_grid
rt_grid = pulse.rt_grid
rdt = pulse.rdt
r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15]
b_weights = [0.245*0.21, 96e-15]
raman = utils.chi3.nl_response_v(rt_grid, rdt, r_weights, b_weights)

mode = pynlo.media.Mode(v_grid, beta, g3_v=g3, rv_grid=rv_grid, r3_v=raman, z=0.0)


# %% Model

model = pynlo.model.SM_UPE(pulse, mode)

#---- Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(n=20, local_error=local_error)


# %% Simulate

pulse_out, z, a_t, a_v = model.simulate(
    length, dz=dz, local_error=local_error, n_records=100, plot=None)


# %% Plot Results

fig = plt.figure("Simulation Results", clear=True)
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

p_l_dB = 10*np.log10(np.abs(a_v)**2 * model.dv_dl)
p_l_dB -= p_l_dB.max()
ax0.plot(1e9*c/v_grid, p_l_dB[0], color="b")
ax0.plot(1e9*c/v_grid, p_l_dB[-1], color="g")
ax2.pcolormesh(1e9*c/v_grid, 1e3*z, p_l_dB, vmin=-40.0, vmax=0, shading="auto")
ax0.set_ylim(bottom=-50, top=10)
ax2.set_xlabel('Wavelength (nm)')

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
