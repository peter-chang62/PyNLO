"""
This script similates the mode-locking of a figure 9 laser based on the
reflective NALM from Hansel et al. (Menlo):

W. Hänsel et al., Appl. Phys. B 123, 41 (2017).

This one is very similar to the kuse_figure9, the reason for including is to
test the implementation of Jones matrices. No splicing loss is taken into
account, so in this case the rep-rate is 200 MHz with only normal dispersion
gain fiber.
"""

# %% ----- imports
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import clipboard  # for clipboard
import collections
from scipy.constants import c

import pynlo
from edf.re_nlse_joint_5level import EDF
import blit
from edf import edfa
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
n_records = None


def propagate(fiber, pulse, length):
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
    model = fiber.generate_model(pulse, t_shock=None, raman_on=False)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- load dispersion coefficients ------------------------------
polyfit = ER80_4_125_betas().polyfit

# %% ------------- pulse ------------------------------------------------------
# f_r = 100e6
f_r = 200e6

n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 1e-9

t_fwhm = 2e-12
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
dv_dl = pulse.v_grid**2 / c

# %% --------- passive fibers -------------------------------------------------
gamma_pm1550 = 1.2
gamma_edf = 6.5

pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_pm1550 / (W * km)

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
edf.set_beta_from_beta_n(v0, polyfit)  # only gdd
edf.gamma = gamma_edf / (W * km)

# %% ----- figure 9 laser cavity with PBS
beta2_g = polyfit[0]
D_g = -2 * np.pi * c / 1560e-9**2 * beta2_g / ps * nm * km
D_p = 18

# total fiber length to hit rep-rate, accounting for free space section in the
# linear arm
l_t = c / f_r / 1.5

# target total round trip dispersion: D_l -> D_rt
D_rt = 0.8
l_g = -l_t * (D_p - D_rt) / (D_g - D_p)
l_p = l_t - l_g  # passive fiber length

assert np.all(np.array([l_g, l_p]) >= 0)
print(f"normal gain: {l_g}, passive in loop: {l_p}")

# wave plates
# Jones matrix for a wave plate
J_wp = lambda phi: np.array(
    [
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)],
    ]
)

# rotation matrix (faraday rotator)
J_rot = lambda theta: np.array(
    [
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ]
)

wp = lambda theta, phi: J_rot(-theta) @ J_wp(phi) @ J_rot(theta)

# initialize to noise, do this if you want to simulate mode-locking!
pulse.a_t[:] = np.random.uniform(0, 1, size=pulse.n)
pulse.e_p = 0.1e-6 / f_r  # .1 microwats

p_gf = pulse.copy()
p_pf = pulse.copy()
p_out = pulse.copy()
p_kept = pulse.copy()

# pump power
# Pp = 50 * 1e-3
Pp = 200 * 1e-3

loop_count = 0
done = False

# set up plot
fig, ax = plt.subplots(3, 2, num=f"{D_rt} ps/nm/km, {np.round(Pp * 1e3, 3)} mW pump")
ax[0, 0].set_xlabel("wavelength (nm)")
ax[1, 0].set_xlabel("wavelength (nm)")
ax[2, 0].set_xlabel("wavelength (nm)")
ax[0, 1].set_xlabel("time (ps)")
ax[1, 1].set_xlabel("time (ps)")
ax[2, 1].set_xlabel("time (ps)")
while not done:
    # ------------- passive fiber first --------------------------
    # passive fiber
    p_pf.a_t[:] = propagate(pm1550, p_pf, l_p).sim.pulse_out.a_t[:]

    # ----------- gain section ---------------------------------
    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=p_gf,
        p_bck=p_pf,
        edf=edf,
        length=l_g,
        Pp_fwd=Pp,
        Pp_bck=0.0,
        t_shock=None,
        raman_on=False,
        n_records=n_records,
    )
    p_gf.a_t[:] = sim_fwd.pulse_out.a_t[:]
    p_pf.a_t[:] = sim_bck.pulse_out.a_t[:]

    # ------------- passive fiber second --------------------------
    p_gf.a_t[:] = propagate(pm1550, p_gf, l_p).sim.pulse_out.a_t[:]

    # ------------- free space section ----------------------------
    # transmission is: cos(phi_bias / 2) ** 2, reflection is: 1 - cos(phi_bias / 2) ** 2
    phi_bias = -np.pi / 2
    ewp = wp(0, phi_bias / 2)
    hwp = wp(np.pi / 4 / 2, np.pi)

    AT = np.array([p_pf.a_t, p_gf.a_t])

    # apply second half of phase bias
    # then rotate back to axis of the PBS
    AT = ewp @ AT  # at this point, the phase bias is given by np.diff(np.angle(AT))
    AT = hwp @ AT  # at this point, abs(AT) ** 2 will show transmission and reflection

    # PBS
    p_out.a_t[:] = AT[1][:]
    AT[1][:] = 0

    p_kept.a_t[:] = AT[0][:]
    oc_percent = p_out.e_p / (p_out.e_p + p_kept.e_p)

    # rotate back to axis of wave plate
    # then apply first half of phase bias
    AT = hwp.T @ AT
    AT = ewp.T @ AT

    p_pf.a_t[:] = AT[0][:]
    p_gf.a_t[:] = AT[1][:]

    # ------------- update ---------------------------------------
    center = pulse.n // 2
    p_pf.a_t[:] = np.roll(p_pf.a_t, center - p_pf.p_t.argmax())
    p_gf.a_t[:] = np.roll(p_gf.a_t, center - p_gf.p_t.argmax())
    p_out.a_t[:] = np.roll(p_out.a_t, center - p_out.p_t.argmax())
    p_kept.a_t[:] = np.roll(p_kept.a_t, center - p_kept.p_t.argmax())

    # update plot
    if loop_count == 0:
        (l1,) = ax[0, 0].plot(
            p_pf.wl_grid * 1e9,
            p_pf.p_v / p_pf.p_v.max() * dv_dl,
            animated=True,
        )
        (l2,) = ax[0, 1].plot(
            p_pf.t_grid * 1e12,
            p_pf.p_t / p_pf.p_t.max(),
            animated=True,
        )
        (l3,) = ax[1, 0].plot(
            p_gf.wl_grid * 1e9,
            p_gf.p_v / p_gf.p_v.max() * dv_dl,
            animated=True,
        )
        (l4,) = ax[1, 1].plot(
            p_gf.t_grid * 1e12,
            p_gf.p_t / p_gf.p_t.max(),
            animated=True,
        )
        (l5,) = ax[2, 0].plot(
            p_out.wl_grid * 1e9,
            p_out.p_v / p_out.p_v.max() * dv_dl,
            animated=True,
        )
        (l6,) = ax[2, 1].plot(
            p_out.t_grid * 1e12,
            p_out.p_t / p_out.p_t.max(),
            animated=True,
        )
        fr_number = ax[0, 0].annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )
        fig.tight_layout()
        bm = blit.BlitManager(fig.canvas, [l1, l2, l3, l4, l5, l6, fr_number])
        bm.update()
    else:
        l1.set_ydata(p_pf.p_v / p_pf.p_v.max() * dv_dl)
        l2.set_ydata(p_pf.p_t / p_pf.p_t.max())
        l3.set_ydata(p_gf.p_v / p_gf.p_v.max() * dv_dl)
        l4.set_ydata(p_gf.p_t / p_gf.p_t.max())
        l5.set_ydata(p_out.p_v / p_out.p_v.max() * dv_dl)
        l6.set_ydata(p_out.p_t / p_out.p_t.max())
        fr_number.set_text(f"loop #: {loop_count}")
        bm.update()

    loop_count += 1
    print(
        f"pulse passive first: {np.round(p_pf.e_p * 1e9, 3)} nJ, \n "
        f"pulse gain first: {np.round(p_gf.e_p * 1e9, 3)} nJ, \n"
        f"output power: {np.round(p_out.e_p * f_r * 1e3, 3)} mW \n "
        f"output coupler percent: {np.round(oc_percent*100,3)}"
    )
