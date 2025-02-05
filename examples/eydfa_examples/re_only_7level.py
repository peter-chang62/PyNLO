"""
This file calculates the broadband amplification of a 100 femtosecond
pulse. It does not use pynlo, but only the rate equations, and can be used to
sanity check the NLSE sims in the limit that gamma -> 0.
"""

# %% ----- imports
import sys

sys.path.append("../../")
import numpy as np
import matplotlib.pyplot as plt
import pynlo
import clipboard
from scipy.constants import c
from scipy.integrate import odeint
from scipy.constants import h
from eydf.seven_level_ss_eqns import (
    sigma_a_p_Y,
    sigma_e_p_Y,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
    tau_ba,
    R,
    _dn1_dt_func,
    _dn2_dt_func,
    dn3_dt_func,
    _dn4_dt_func,
    dn5_dt_func,
    dna_dt_func,
    dnb_dt_func,
    dPp_dz,
    gain,
)

from edf.utility import crossSection

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% ------------- pulse ------------------------------------------------------
f_r = 1e9
e_p = 100e-3 / f_r

n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
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


# %% -------------------------------------------------------------------------
sigma_p = spl_sigma_a(c / 980e-9)
sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)

n_ion = 55 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm
r_eff = 5.5 * um
a_eff = np.pi * r_eff**2
nu_p = c / 980e-9

n_ion_Y = n_ion * 10

overlap_p = 5.5**2 / 65**2
overlap_s = 1.0


def dn_dt(X, t, P_p, P_s):
    n1, n2, n3, n4, n5, na, nb = X

    sum_a = overlap_s * P_s * sigma_a / (h * pulse.v_grid * a_eff)
    sum_e = overlap_s * P_s * sigma_e / (h * pulse.v_grid * a_eff)
    sum_a = np.sum(sum_a)
    sum_e = np.sum(sum_e)

    dn1_dt = _dn1_dt_func(
        a_eff,
        overlap_p,
        nu_p,
        P_p,
        n1,
        n2,
        n3,
        na,
        nb,
        n_ion_Y,
        sum_a,
        sum_e,
        sigma_p,
        xi_p,
        tau_21,
        R,
    )

    dn2_dt = _dn2_dt_func(
        a_eff,
        n1,
        n2,
        n3,
        sum_a,
        sum_e,
        eps_s,
        tau_21,
        tau_32,
    )

    dn3_dt = dn3_dt_func(
        a_eff,
        overlap_p,
        nu_p,
        P_p,
        n1,
        n3,
        n4,
        na,
        nb,
        n_ion_Y,
        sigma_p,
        xi_p,
        eps_p,
        tau_32,
        tau_43,
        R,
    )

    dn4_dt = _dn4_dt_func(
        a_eff,
        n2,
        n4,
        n5,
        sum_a,
        eps_s,
        tau_43,
        tau_54,
    )

    dn5_dt = dn5_dt_func(
        a_eff,
        overlap_p,
        nu_p,
        P_p,
        n3,
        n5,
        sigma_p,
        eps_p,
        tau_54,
    )

    dna_dt = dna_dt_func(
        a_eff,
        overlap_p,
        nu_p,
        P_p,
        n1,
        n3,
        na,
        nb,
        n_ion,
        sigma_a_p_Y,
        sigma_e_p_Y,
        tau_ba,
        R,
    )

    dnb_dt = dnb_dt_func(
        a_eff,
        overlap_p,
        nu_p,
        P_p,
        n1,
        n3,
        na,
        nb,
        n_ion,
        sigma_a_p_Y,
        sigma_e_p_Y,
        tau_ba,
        R,
    )
    return np.array(
        [
            dn1_dt,
            dn2_dt,
            dn3_dt,
            dn4_dt,
            dn5_dt,
            dna_dt,
            dnb_dt,
        ]
    )


def func(X, z, output="deriv"):
    P_p = X[0]
    P_v = X[1:]

    t = np.linspace(0, 0.1, 2)
    X_0 = np.array([1, 0, 0, 0, 0, 1, 0])
    sol = odeint(dn_dt, X_0, t, args=(P_p, P_v))

    n1, n2, n3, n4, n5, na, nb = sol[-1]

    dP_p = dPp_dz(
        overlap_p,
        P_p,
        n1,
        n3,
        na,
        nb,
        n_ion,
        n_ion_Y,
        sigma_p,
        sigma_a_p_Y,
        sigma_e_p_Y,
        xi_p,
        eps_p,
    )

    dP_v = gain(
        overlap_s,
        n1,
        n2,
        n_ion,
        sigma_a,
        sigma_e,
        eps_s,
    ) * P_v

    print(z)

    if output == "deriv":
        return np.hstack((dP_p, dP_v))
    else:
        return sol[-1]


Pp_0 = 18
Pv_0 = pulse.p_v.copy() * pulse.dv * f_r
length = 6

X_0 = np.hstack((Pp_0, Pv_0))
z = np.linspace(0, length, 1000)
sol = odeint(func, X_0, z)

sol_Pp = sol[:, 0]
sol_Pv = sol[:, 1:]
sol_Ps = np.sum(sol_Pv, axis=1)

n1 = np.zeros(z[::10].size)
n2 = np.zeros(z[::10].size)
n3 = np.zeros(z[::10].size)
n4 = np.zeros(z[::10].size)
n5 = np.zeros(z[::10].size)
na = np.zeros(z[::10].size)
nb = np.zeros(z[::10].size)
for n, (pp, pv) in enumerate(zip(sol_Pp[::10], sol_Pv[::10])):
    inversion = func(np.hstack((pp, pv)), z[::10][n], output="n")
    n1[n] = inversion[0]
    n2[n] = inversion[1]
    n3[n] = inversion[2]
    n4[n] = inversion[3]
    n5[n] = inversion[4]
    na[n] = inversion[5]
    nb[n] = inversion[6]

# %% ----------------------------- plot results! ------------------------------
fig = plt.figure(
    num="5-level rate equation for 250 fs pulse", figsize=np.array([11.16, 5.21])
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
(line_11,) = ax1.plot(z, sol_Pp, label="pump", linewidth=2)
(line_12,) = ax1.plot(z, sol_Ps, label="signal", linewidth=2)
ax1.grid()
ax1.legend(loc="best")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

(line_21,) = ax2.plot(z[::10], n1, label="n1", linewidth=2)
(line_22,) = ax2.plot(z[::10], n2, label="n2", linewidth=2)
(line_23,) = ax2.plot(z[::10], n3, label="n3", linewidth=2)
(line_24,) = ax2.plot(z[::10], n4, label="n4", linewidth=2)
(line_25,) = ax2.plot(z[::10], n5, label="n5", linewidth=2)
(line_26,) = ax2.plot(z[::10], na, label="na", linewidth=2)
(line_27,) = ax2.plot(z[::10], nb, label="nb", linewidth=2)
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")

fig.tight_layout()
