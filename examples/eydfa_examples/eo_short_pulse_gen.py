# %% -----
import numpy as np
import pynlo
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c
import collections
from edf.utility import crossSection, ER80_4_125_betas
from eydf.re_nlse_joint_7level import EYDF
from eydf import eydfa
from edf.re_nlse_joint_5level import EDF
from edf import edfa as edfa_core

ps = 1e-12
nm = 1e-9
um = 1e-6
m = 1
km = 1e3
W = 1e3

output = collections.namedtuple("output", ["model", "sim"])


# %% --------------------------------------------------------------------------
def propagate(fiber, pulse, length, n_records=100):
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
    model = fiber.generate_model(pulse, t_shock=None, raman_on=True)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% --------------------------------------------------------------------------
v0 = c / 1550e-9
v_min = c / 1600e-9
v_max = c / 1500e-9
V = v_max - v_min
dv = 20e9
N = int(np.round(V / dv))
min_time_window = 50e-12  # comb resolution

f_r = 20e9
e_p = 1e-3 / f_r
t_fwhm = 50e-15  # useless parameter

pulse = pynlo.light.Pulse.Sech(
    N,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
    alias=2,
)
dv_dl = pulse.v_grid**2 / c

# %% ----------- send through phase modulators --------------------------------
omega = 20e9 * 2 * np.pi
w0 = pulse.v0 * 2 * np.pi
w_ref = pulse.v_ref * 2 * np.pi
t = pulse.t_grid
K = 36

phi = (w0 - w_ref) * t + K * np.sin(omega * t)
a_t = np.exp(1j * phi)
pulse.a_t[:] = a_t
pulse.e_p = e_p

# %% ----- send through the Intensity modulator -------------------------------
phi_del = 180 * np.pi / 180
i_mod = np.sin(omega * t + phi_del) / 2 + 0.5
pulse.p_t[:] *= i_mod

# center the pulse (remove linear phase)
center = pulse.n // 2
roll = center - pulse.p_t.argmax()
pulse.a_t[:] = np.roll(pulse.a_t[:], roll)

# %% ----- fit the phase to determine the chirp -------------------------------
v_width = pulse.v_width()
v_width = max([v_width.eqv, v_width.rms, v_width.fwhm])
xlim = pulse.v0 - v_width / 2, pulse.v0 + v_width / 2
(idx,) = np.logical_and(xlim[0] < pulse.v_grid, pulse.v_grid < xlim[1]).nonzero()

p = np.unwrap(pulse.phi_v[idx])
p -= p.min()
polyfit = np.polyfit((pulse.v_grid[idx] - pulse.v0) * 2 * np.pi, p, deg=3)
betas = np.asarray([i * np.math.factorial(n) for n, i in enumerate(polyfit[::-1])])
Dm = betas[2] * -2 * np.pi * c / 1550e-9**2
Dm /= ps / nm
print(f"approximate chirp is {Dm} / (ps nm)")

# %% ----- pulse recompression ------------------------------------------------
dcf = pynlo.materials.SilicaFiber()
dcf.set_beta_from_D_n(1550e-9, 100 * ps / nm / km, 0)
dcf.gamma = 0 / (W * m)
sim_dcf = propagate(dcf, pulse, Dm * 10, n_records=100).sim

# p = pulse.copy()
# t_width = np.zeros(sim_dcf.z.size, dtype=float)
# for n, a_t in enumerate(sim_dcf.a_t):
#     p.a_t[:] = a_t[:]
#     t_width[n] = p.t_width().fwhm
# idx = t_width.argmin()
# Z = sim_dcf.z[idx]
# p.a_t[:] = sim_dcf.a_t[idx]
# print("length passive compression: ", np.round(sim_dcf.z[idx], 3))

# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- gamma values ----------------------------------------------
gamma_a = 1 / (W * m)
gamma_n = 6.5 / (W * m)

# %% -------------- load dispersion coefficients ------------------------------
polyfit_n = ER80_4_125_betas().polyfit

# %% ------------ active fiber ------------------------------------------------
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_n)  # only gdd
edf.gamma = gamma_n

# %% ---------- edfa ----------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa_core.amplify(
    p_fwd=sim_dcf.pulse_out,
    p_bck=None,
    edf=edf,
    length=1.0,
    Pp_fwd=0,
    Pp_bck=1,
    n_records=100,
    raman_on=False,
)
sim_edfa = sim_fwd

# %% --------- after edfa, go through some passive fibers ---------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_a

edf_passive = pynlo.materials.SilicaFiber()
edf_passive.set_beta_from_beta_n(v0, polyfit_n)
edf_passive.gamma = gamma_n

hnlf_a = pynlo.materials.SilicaFiber()
hnlf_a.load_fiber_from_dict(pynlo.materials.hnlf_5p7)

D_hnlf = -1
polyfit_hnlf = np.array([-(1550e-9**2) / (2 * np.pi * c) * (D_hnlf * ps / nm / km)])
hnlf_n = pynlo.materials.SilicaFiber()
hnlf_n.set_beta_from_beta_n(v0, polyfit_hnlf)
hnlf_n.gamma = 10 / (W * m)

# %% ------------ passive fiber -----------------------------------------------
sim_pm1550 = propagate(pm1550, sim_edfa.pulse_out, 2).sim

# %% ------------ eydf doped fiber --------------------------------------------
n_ion = 55 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm
r_eff = 5.5 * um
a_eff = np.pi * r_eff**2
n_ion_Y = n_ion * 10

r_mm = 65 * um
a_mm = np.pi * r_mm**2
overlap_p = a_eff / a_mm
overlap_s = 1.0

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

length = 6

eydf = EYDF(
    f_r=f_r,
    overlap_p=overlap_p,
    overlap_s=1.0,
    n_ion=n_ion,
    n_ion_Y=n_ion_Y,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
D_g = 18
polyfit_a = np.array([-(1550e-9**2) / (2 * np.pi * c) * (D_g * ps / nm / km)])
eydf.set_beta_from_beta_n(v0, polyfit_a)
beta_n = eydf._beta(pulse.v_grid)
eydf.gamma = gamma_a

# %% ----------- eydf ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = eydfa.amplify(
    p_fwd=sim_pm1550.pulse_out,
    p_bck=None,
    eydf=eydf,
    length=length,
    Pp_fwd=35,
    Pp_bck=0,
    n_records=100,
    raman_on=True,
)
sim_eydfa = sim_fwd

# %% --------------------------------------------------------------------------
v_min = c / 2500e-9
v_max = c / 1000e-9
min_time_window = 50e-12
e_p = sim_eydfa.pulse_out.e_p
p_hnlf = pynlo.light.Pulse.Sech(
    256,
    v_min,
    v_max,
    v0,
    e_p,
    50e-15,
    min_time_window,
)
p_hnlf.import_p_v(
    pulse.v_grid,
    sim_eydfa.pulse_out.p_v,
    phi_v=np.unwrap(sim_eydfa.pulse_out.phi_v),
)

# %% --------------------------------------------------------------------------
sim_a_1 = propagate(pm1550, p_hnlf, 20.0, n_records=100).sim
p = p_hnlf.copy()
t_width = np.zeros(sim_a_1.z.size, dtype=float)
for n, a_t in enumerate(sim_a_1.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_1 = sim_a_1.z[idx]
p.a_t[:] = sim_a_1.a_t[idx]
print("length passive compression: ", np.round(sim_a_1.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_1 = propagate(edf_passive, p, 10.0, n_records=100).sim
for n, a_t in enumerate(sim_n_1.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_1 = sim_n_1.z[idx]
p.a_t[:] = sim_n_1.a_t[idx]
print("length passive 1 normal: ", np.round(sim_n_1.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_2 = propagate(pm1550, p, 5.0, n_records=100).sim
for n, a_t in enumerate(sim_a_2.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_2 = sim_a_2.z[idx]
p.a_t[:] = sim_a_2.a_t[idx]
print("length passive 1 anomalous: ", np.round(sim_a_2.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_2 = propagate(edf_passive, p, 1.0, n_records=100).sim
for n, a_t in enumerate(sim_n_2.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_2 = sim_n_2.z[idx]
p.a_t[:] = sim_n_2.a_t[idx]
print("length passive 2 normal: ", np.round(sim_n_2.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_3 = propagate(pm1550, p, 5.0, n_records=100).sim
for n, a_t in enumerate(sim_a_3.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_3 = sim_a_3.z[idx]
p.a_t[:] = sim_a_3.a_t[idx]
print("length passive 2 anomalous: ", np.round(sim_a_3.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_3 = propagate(edf_passive, p, 1.0, n_records=100).sim
for n, a_t in enumerate(sim_n_3.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_3 = sim_n_3.z[idx]
p.a_t[:] = sim_n_3.a_t[idx]
print("length passive 3 normal: ", np.round(sim_n_3.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_4 = propagate(pm1550, p, 1.0, n_records=100).sim
for n, a_t in enumerate(sim_a_4.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_4 = sim_a_4.z[idx]
p.a_t[:] = sim_a_4.a_t[idx]
print("length passive 3 anomalous: ", np.round(sim_a_4.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_4 = propagate(edf_passive, p, 1.0, n_records=100).sim
for n, a_t in enumerate(sim_n_4.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_4 = sim_n_4.z[idx]
p.a_t[:] = sim_n_4.a_t[idx]
print("length passive 4 normal: ", np.round(sim_n_4.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_5 = propagate(pm1550, p, 1.0, n_records=100).sim
for n, a_t in enumerate(sim_a_5.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_5 = sim_a_5.z[idx]
p.a_t[:] = sim_a_5.a_t[idx]
print("length passive 4 anomalous: ", np.round(sim_a_5.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_5 = propagate(hnlf_n, p, 3.0, n_records=100).sim
for n, a_t in enumerate(sim_n_5.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_5 = sim_n_5.z[idx]
p.a_t[:] = sim_n_5.a_t[idx]
print("length 1 HNLF normal: ", np.round(sim_n_5.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_6 = propagate(hnlf_a, p, 0.5, n_records=100).sim
for n, a_t in enumerate(sim_a_6.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_6 = sim_a_6.z[idx]
p.a_t[:] = sim_a_6.a_t[idx]
print("length 1 HNLF anomalous: ", np.round(sim_a_6.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_6 = propagate(hnlf_n, p, 0.7, n_records=100).sim
for n, a_t in enumerate(sim_n_6.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_6 = sim_n_6.z[idx]
p.a_t[:] = sim_n_6.a_t[idx]
print("length 2 HNLF normal: ", np.round(sim_n_6.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_7 = propagate(hnlf_a, p, 0.2, n_records=100).sim
for n, a_t in enumerate(sim_a_7.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_7 = sim_a_7.z[idx]
p.a_t[:] = sim_a_7.a_t[idx]
print("length 2 HNLF anomalous: ", np.round(sim_a_7.z[idx], 3))

# %% --------------------------------------------------------------------------
t_width_0 = p.t_width().eqv
sim_n_7 = propagate(hnlf_n, p, 0.7, n_records=100).sim
for n, a_t in enumerate(sim_n_7.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = abs(t_width - t_width_0 * 2.0).argmin()
Z_n_7 = sim_n_7.z[idx]
p.a_t[:] = sim_n_7.a_t[idx]
print("length 3 HNLF normal: ", np.round(sim_n_7.z[idx], 3))

# %% --------------------------------------------------------------------------
sim_a_8 = propagate(hnlf_a, p, 0.05, n_records=100).sim
for n, a_t in enumerate(sim_a_8.a_t):
    p.a_t[:] = a_t[:]
    t_width[n] = p.t_width().eqv
idx = t_width.argmin()
Z_a_8 = sim_a_8.z[idx]
p.a_t[:] = sim_a_8.a_t[idx]
print("length 3 HNLF anomalous: ", np.round(sim_a_8.z[idx], 3))

# %% ------ plot results ------------------------------------------------------
dv_dl = p.v_grid**2 / c

figsize = np.array([6.65, 7.86])

fig, ax = plt.subplots(8, 2, figsize=figsize)

# ----- anomalous dispersion ----------------------------------------
p.a_t[:] = sim_a_1.a_t[abs(Z_a_1 - sim_a_1.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[0, 0].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_a_2.a_t[abs(Z_a_2 - sim_a_2.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[1, 0].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_a_3.a_t[abs(Z_a_3 - sim_a_3.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[2, 0].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_a_4.a_t[abs(Z_a_4 - sim_a_4.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[3, 0].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_a_5.a_t[abs(Z_a_5 - sim_a_5.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[4, 0].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_a_6.a_t[abs(Z_a_6 - sim_a_6.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[5, 0].plot(p.wl_grid * 1e9, p_wl, "C1")

p.a_t[:] = sim_a_7.a_t[abs(Z_a_7 - sim_a_7.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[6, 0].plot(p.wl_grid * 1e9, p_wl, "C1")

p.a_t[:] = sim_a_8.a_t[abs(Z_a_8 - sim_a_8.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[7, 0].plot(p.wl_grid * 1e9, p_wl, "C1")

# ----- normal dispersion ----------------------------------------
p.a_t[:] = sim_n_1.a_t[abs(Z_n_1 - sim_n_1.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[0, 1].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_n_2.a_t[abs(Z_n_2 - sim_n_2.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[1, 1].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_n_3.a_t[abs(Z_n_3 - sim_n_3.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[2, 1].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_n_4.a_t[abs(Z_n_4 - sim_n_4.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[3, 1].plot(p.wl_grid * 1e9, p_wl)

p.a_t[:] = sim_n_5.a_t[abs(Z_n_5 - sim_n_5.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[4, 1].plot(p.wl_grid * 1e9, p_wl, "C1")

p.a_t[:] = sim_n_6.a_t[abs(Z_n_6 - sim_n_6.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[5, 1].plot(p.wl_grid * 1e9, p_wl, "C1")

p.a_t[:] = sim_n_7.a_t[abs(Z_n_7 - sim_n_7.z).argmin()]
p_wl = p.p_v * dv_dl
p_wl = 10 * np.log10(p_wl)
p_wl -= p_wl.max()
ax[6, 1].plot(p.wl_grid * 1e9, p_wl, "C1")

ax[7, 1].axis(False)

[i.set_ylim(-50, 10) for i in ax.flatten()]
fig.tight_layout()

# %% ------ plot results ------------------------------------------------------
fig, ax = plt.subplots(8, 2, figsize=figsize)

center = p_hnlf.n // 2

# ----- anomalous dispersion ----------------------------------------
p.a_t[:] = sim_a_1.a_t[abs(Z_a_1 - sim_a_1.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[0, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_a_2.a_t[abs(Z_a_2 - sim_a_2.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[1, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_a_3.a_t[abs(Z_a_3 - sim_a_3.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[2, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_a_4.a_t[abs(Z_a_4 - sim_a_4.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[3, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_a_5.a_t[abs(Z_a_5 - sim_a_5.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[4, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_a_6.a_t[abs(Z_a_6 - sim_a_6.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[5, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

p.a_t[:] = sim_a_7.a_t[abs(Z_a_7 - sim_a_7.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[6, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

p.a_t[:] = sim_a_8.a_t[abs(Z_a_8 - sim_a_8.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[7, 0].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

# ----- normal dispersion ----------------------------------------
p.a_t[:] = sim_n_1.a_t[abs(Z_n_1 - sim_n_1.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[0, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_n_2.a_t[abs(Z_n_2 - sim_n_2.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[1, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_n_3.a_t[abs(Z_n_3 - sim_n_3.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[2, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_n_4.a_t[abs(Z_n_4 - sim_n_4.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[3, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max())

p.a_t[:] = sim_n_5.a_t[abs(Z_n_5 - sim_n_5.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[4, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

p.a_t[:] = sim_n_6.a_t[abs(Z_n_6 - sim_n_6.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[5, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

p.a_t[:] = sim_n_7.a_t[abs(Z_n_7 - sim_n_7.z).argmin()]
roll = center - p.p_t.argmax()
p.a_t[:] = np.roll(p.a_t[:], roll)
ax[6, 1].plot(p.t_grid * 1e12, p.p_t / p.p_t.max(), "C1")

ax[7, 1].axis(False)

[i.set_xlim(-2, 2) for i in ax.flatten()]
[i.set_ylim(-0.2, 1.2) for i in ax.flatten()]
fig.tight_layout()

# %% ------ plot results ------------------------------------------------------
sim_dcf.plot("wvl", num="pulse compression")
sim_edfa.plot("wvl", num="EDFA pre-amp")
sim_eydfa.plot("wvl", num="EYDF power-amp")
