# -*- coding: utf-8 -*-
"""
These examples demonstrate several effects due to the dynamics of second- and
higher-order solitons: self-compression and oscillation at the soliton period,
dispersive wave generation, and Raman-induced frequency shifts. The simulations
use the nonlinear Schrödinger equation (NLSE) and are designed to highlight
processes critical to supercontinuum generation in the anomalous dispersion
regime.

"""

# %% Imports
import numpy as np
from scipy.constants import pi
from matplotlib import pyplot as plt
import pynlo


# %% Pulse Properties
"""
Solitons have a squared hyperbolic secant pulse shape and are defined by a
soliton number `N`, given by the square root of the ratio of the dispersion and
nonlinear length scales (``N=(L_D/L_NL)**0.5``). In the following examples we
use a third-order soliton with a characteristic pulse length `T0` of 50 fs.

Notes
-----
The simulations demonstrate both unperturbed and perturbed soliton dynamics.
For accuracy, there must be no additional perturbations due to limitations of
the underlying time and frequency grids. This means that the frequency span
must be wide enough such that spectral components do not significantly alias as
the soliton breathes, and the time window must be long enough such that
dispersed spectral components do not wrap around and significantly interfere
with the main pulse.
"""
# ---- Soliton Parameters
N = 3  # soliton number

T0 = 50e-15  # 50 fs
gamma = 1  # 1 / W * m
beta2 = -10 * 1e-12**2 / 1e3  # -10 ps**2 / km

# ---- Pulse Parameters
n_points = 2**11
v_min = 100e12  # 100 THz
v_max = 500e12  # 500 THz
v0 = 300e12  # 300 THz

e_p = N**2 / (gamma / np.abs(beta2) * T0 / 2)
t_fwhm = np.arccosh(2**0.5) * 2 * T0
time_window = 10e-12

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm, time_window)

# ---- Length Scales
L_D = T0**2 / np.abs(beta2)
L_NL = 2 * T0 / (gamma * e_p)
L_S = pi / 2 * L_D
L_C = 0.5 * L_S / (N - 1)  # approximate, less accurate at large N

print("Soliton Number \t\t= {:.2g}".format((L_D / L_NL) ** 0.5))
print("Dispersion Length \t= {:.3g} m".format(L_D))
print("Nonlinear Length \t= {:.3g} m".format(L_NL))
print("Soliton Period \t\t= {:.3g} m".format(L_S))
print("Compression Length \t= {:.3g} m".format(L_C))

# %% Unperturbed Soliton Dynamics
"""
The first example demonstrates the spectral and temporal oscillation of an
unperturbed soliton. Fundamental solitons (soliton number ``N=1``) do not
change with propagation distance. However, for second- and higher-order
solitons, the pulse shape and spectra oscillate over the soliton period `L_S`,
a function of the dispersion length scale `L_D`. The compression length scale
`L_C`, the approximate distance at which a soliton reaches its shortest
duration and widest spectral extent, is a function of both the dispersion
length `L_D` and the soliton number `N`.

Integer-value solitons function as nonlinear attractors for non-integer
solitons and for other pulse shapes as well, and in the anomalous dispersion
regime all pulses will evolve towards the nearest integer soliton solution.
This evolution requires the shedding of excess radiation in the form of
dispersive waves, quasi-continuum radiation that is not phase-locked to the
soliton and that spreads apart with propagation distance.

This simulation extends over two soliton periods, and after each soliton period
the pulse returns to its original shape. The input and output spectra are shown
along with the spectra from the first compression point. For more details on
optical solitons, see chapter 5.2 of Agrawal.

References
----------
Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
 CA;: Academic Press; 2019.
 https://doi.org/10.1016/B978-0-12-817042-7.00009-9

"""
# ---- Mode Properties
length = L_S * 2

fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(pulse.v0, [beta2])
fiber.gamma = gamma

# ---- Model
model = fiber.generate_model(pulse, t_shock=None, raman_on=False, method="nlse")

# Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(local_error=local_error)

# ----- Simulate
sim = model.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)

# ----- Plot Results
sim.plot("frq", num="Soliton Dynamics")

# %% Dispersive Wave Generation
"""
For the second example we perturb the soliton with third-order dispersion.
Perturbations cause second and higher-order solitons to break apart into
fundamental solitons and dispersive waves. Soliton fission typically occurs
near what would be the first compression point (`L_C`) of the unperturbed
soliton as the perturbation is strongest when the spectral width is at its
greatest extent.

A soliton phase matches to dispersive waves wherever the phase coefficient of
the soliton equals that of continuous radiation::

    beta[w] = beta[w0] + beta1[w0]*(w - w0) + 0.5*gamma*P[z]

The left-hand side of the equation represents the phase coefficient of the
dispersive radiation and the right-hand side represents the phase coefficient
of the soliton. The phase coefficient of the soliton depends on the local peak
power `P[z]`. All frequencies in the formula (`w`) are angular, with `w0` being
the center frequency of the pulse. Ignoring the peak power, the phase matching
condition predicts dispersive wave generation at around 350 THz for the mode
parameters defined below.

This simulation extends over two soliton periods, but after fission the pulse
does not return to its original shape. Instead, the fission process locks in
the spectra of a fundamental soliton and the dispersive waves. Supercontinua
generated with dispersive waves are typically the most filled in or uniform
just before the point of soliton fission. For more details on soliton fission
and dispersive wave generation, see chapter 12.1 of Agrawal.

References
----------
Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
 CA;: Academic Press; 2019.
 https://doi.org/10.1016/B978-0-12-817042-7.00009-9

"""
# ---- Mode Properties
length = L_S * 2
beta_n = 2 * [0]
beta_n[0] = beta2
beta_n[1] = 0.1 * 1e-12**3 / 1e3  # 0.1 ps**3/km

fiber.set_beta_from_beta_n(pulse.v0, beta_n)

# ------ Model
model = fiber.generate_model(pulse, t_shock=None, raman_on=False, method="nlse")

# Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(local_error=local_error)

# ----- Simulate
sim = model.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)

# ----- Plot Results
sim.plot("frq", num="Dispersive Wave Generation")

# %% Raman-Induced Frequency Shift
"""
The third example is perturbed by the Raman effect of silica-based optical
fibers. Like higher-order dispersion, the Raman effect can also induce soliton
fission. This occurs through the Raman-induced transfer of energy from the
high-frequency side of the pulse to the low-frequency side, which changes the
effective center frequency of the soliton. To first order this frequency shift
is linear with propagation distance, and the magnitude of the change scales
with the fourth power of the soliton's spectral width, i.e. a shorter pulse
will shift more quickly than a longer pulse.

This simulation extends over two soliton periods, but as with all other sources
of perturbation the pulse does not return to its original shape after soliton
fission. Instead, the Raman soliton continuously shifts to lower frequency,
leaving behind residual fundamental solitons and dispersive waves at the pump
frequency. The Raman-induced frequency shift can be adjusted with small changes
to the waveguide length or pump power, which is useful for fine-tuning the
generation of octave-spanning spectra for carrier-envelope offset detection and
frequency comb stabilization. For more details on intrapulse Raman scattering
see chapter 12.2 of Agrawal.

References
----------
Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
 CA;: Academic Press; 2019.
 https://doi.org/10.1016/B978-0-12-817042-7.00009-9

"""
# ---- Mode Properties
length = L_S * 2

# Phase Coefficient
beta_n = [beta2]

fiber.set_beta_from_beta_n(pulse.v0, beta_n)

# Raman effect
r_weights = [0.245 * (1 - 0.21), 12.2e-15, 32e-15]  # resonant contribution
b_weights = [0.245 * 0.21, 96e-15]  # boson contribution
fiber.r_weights = r_weights
fiber.b_weights = b_weights

# ----- Model
model = fiber.generate_model(pulse, t_shock=None, raman_on=True, method="nlse")

# ----- Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(local_error=local_error)

# ----- Simulate
sim = model.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)

# ----- Plot Results
sim.plot("frq", num="Raman-Induced Frequency Shift")
