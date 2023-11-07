"""
The following example illustrates intra-pulse DFG starting from supercontinuum
generation in fiber.

A 200 fs pulse is sent through 17cm of pm-1550, followed by <1cm of anomalous
dispersion HNLF. The length is chosen to meet the pulse at the soliton fission
point, which is done by selecting the propagation distance at which the pulse
duration is a minimum.

The short pulse after the HNLF is then sent into bulk PPLN for intra-pulse
DFG.
"""

# %%
import pynlo
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import clipboard as cr


# %% ----- Pulse Properties
v_min = c / 6500e-9
v_max = c / 450e-9
time_window = 10e-12
npts = 2**12
v0 = c / 1550e-9
t_fwhm = 200e-15
e_p = 4.0e-9

pulse = pynlo.light.Pulse.Sech(npts, v_min, v_max, v0, e_p, t_fwhm, time_window)

# %% ----- Propagation through PM-1550
length = 17e-2
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)

model = pm1550.generate_model(pulse, t_shock="auto", raman_on=True, method="nlse")

dz = model.estimate_step_size()
sim_pm1550 = model.simulate(length, dz=dz, n_records=100)

# %% ----- Propagation through anomalous dispersion HNLF from OFS
length = 2e-2
hnlf = pynlo.materials.SilicaFiber()
hnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)

model = hnlf.generate_model(
    sim_pm1550.pulse_out, t_shock="auto", raman_on=True, method="nlse"
)

dz = model.estimate_step_size()
sim_hnlf = model.simulate(length, dz=dz, n_records=100)

# %% ----- find the minimum pulse duration inside the HNLF
pulse_out = pulse.copy()
T_WIDTH = np.zeros(len(sim_hnlf.z))
for n, a_v in enumerate(sim_hnlf.a_v):
    pulse_out.a_v = a_v
    t_width = pulse_out.t_width()
    T_WIDTH[n] = t_width.eqv

idx = T_WIDTH.argmin()
pulse_out.a_v = sim_hnlf.a_v[idx]
z_cut = sim_hnlf.z[idx]

# %% ----- Propagation through PPLN
length = 1e-3
a_eff = np.pi * 15e-6**2
p0 = 27.5e-6
dk = 2 * np.pi / p0
g2_inv, *_ = pynlo.utility.chi2.domain_inversions(length, dk)
ppln = pynlo.materials.MgLN(T=24.5, axis="e")

model = ppln.generate_model(
    pulse_out, a_eff, length, g2_inv=g2_inv, beta=None, is_gaussian_beam=True
)

dz = model.estimate_step_size()
sim_ppln = model.simulate(length, dz=dz, n_records=100)

# %% ----- Plotting
sim_pm1550.plot("wvl", num="PM-1550")
fig, ax = sim_hnlf.plot("wvl", num="HNLF")
ax[1, 0].axhline(z_cut * 1e3, color="k", linestyle="--")
ax[1, 1].axhline(z_cut * 1e3, color="k", linestyle="--")
sim_ppln.plot("wvl", num=f"PPLN poled at {p0 * 1e6} um")
