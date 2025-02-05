# %% ----- imports
import sys

sys.path.append("../")
from scipy.constants import c
import clipboard
from edf.re_nlse_joint_5level import EDF
import pynlo
import numpy as np
import matplotlib.pyplot as plt
from edf import edfa
import collections
from edf.utility import crossSection, ER80_4_125_betas


ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])


def propagate(fiber, pulse, length, n_records=None):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- load dispersion coefficients ------------------------------
polyfit_n = ER80_4_125_betas().polyfit

gamma_n = 6.5 / (W * km)
gamma_a = 1.2 / (W * km)

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)
loss_mat = 10 ** (-1 / 10)

f_r = 200e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 35e-3 / 2 / f_r

t_fwhm = 100e-15
min_time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
    alias=2,
)
dv_dl = pulse.v_grid**2 / c  # J / Hz -> J / m

# %% ---------- optional passive fiber ----------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_a / (W * km)

length_pm1550 = 1.119
# ignore numpy error if length = 0.0, it occurs when n_records is not None and
# propagation length is 0, the output pulse is still correct
model_pm1550, sim_pm1550 = propagate(pm1550, pulse, length_pm1550)
pulse_pm1550 = sim_pm1550.pulse_out

# %% ------------ active fiber ------------------------------------------------
r_eff_n = 3.06 * um / 2
r_eff_a = 8.05 * um / 2
a_eff_n = np.pi * r_eff_n**2
a_eff_a = np.pi * r_eff_a**2
n_ion_n = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)
n_ion_a = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

length = 1.5

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion_n,
    a_eff=a_eff_n,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_n)
beta_n = edf._beta(pulse.v_grid)
edf.gamma = gamma_n

# %% ----------- edfa ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=pulse_pm1550,
    p_bck=None,
    edf=edf,
    length=length,
    Pp_fwd=2 * loss_ins * loss_spl,
    Pp_bck=2 * loss_ins * loss_spl,
    n_records=100,
)
sim = sim_fwd

# %% ----------- plot results -------------------------------------------------
sol_Pp = sim.Pp
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n

fig = plt.figure(
    num=f"power evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
    figsize=np.array([11.16, 5.21]),
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(z, sol_Pp, label="pump", linewidth=2)
ax1.plot(z, sol_Ps * loss_ins * loss_spl, label="signal", linewidth=2)
ax1.grid()
ax1.legend(loc="upper left")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

ax2.plot(z, n1, label="n1", linewidth=2)
ax2.plot(z, n2, label="n2", linewidth=2)
ax2.plot(z, n3, label="n3", linewidth=2)
ax2.plot(z, n4, label="n4", linewidth=2)
ax2.plot(z, n5, label="n5", linewidth=2)
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")
fig.tight_layout()

sim.plot(
    "wvl",
    num=f"spectral evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
)
