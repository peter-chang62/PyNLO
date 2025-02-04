# -----------------------------------------------------------------------------
# The rate equations for Ytterbium coupling are not analytic; the cross terms
# mean they are no longer linear!

# -----------------------------------------------------------------------------

# %% ----- imports
import numpy as np
from scipy.constants import h

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
cm = 1e-2
km = 1e3
W = 1.0

# I think the Er stimulated emission and ESA at 980 nm -> 0 in the case of Yb
# co-doping, I don't see it in any EYDF publications (Han et al. 2010, Dong et
# al. 2023)
xi_p = 0  # sigma_31 / sigma_13: ratio of sigma_e to sigma_a at 980 nm
eps_p = 0  # sigma_35 / sigma_13: ratio of sigma_esa to sigma_a at 980 nm
eps_s = 0.15  # sigma_24/sigma_12: ratio of sigma_esa to sigma_a for the signal

# Q. Han, J. Ning, and Z. Sheng, "Numerical Investigation of the ASE and Power
# Scaling of Cladding-Pumped Er–Yb Codoped Fiber Amplifiers," IEEE J. Quantum
# Electron. 46, 1535–1541 (2010).
# lifetimes
tau_21 = 10 * ms  # erbium excited state lifetime
tau_ba = 1.5 * ms  # ytterbium excited state lifetime

# much faster than barmenkov et al., er pump excited state is basically unpopulated
tau_32 = 1 * ns

# the following lifetimes are taken to copy the results of Han et al. they
# basically make the esa and stimulated emission cross-section for the Er
# 980nm band level irrelevant
tau_43 = 1.0 * ns  # er signal esa level basically unpopulated
tau_54 = 1.0 * ns  # er pump esa level basically unpopulated

# ----------------------------

# cross-relaxation between Yb <-> Er
R = 2.371e-22  # m^3/s
sigma_a_p_Y = 1.2e-24  # m^2
sigma_e_p_Y = 1.2e-24  # m^2


def dn1_dt(
    n1,
    n2,
    n3,
    na,
    nb,
    Nyb,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    tau_21,
    R,
):
    return (
        -sigma_13 * n1
        - sigma_12 * n1
        + sigma_21 * n2
        + sigma_31 * n3
        + n2 / tau_21
        - R * nb * Nyb * n1
        + R * na * Nyb * n3
    )


def dn2_dt(
    n1,
    n2,
    n3,
    sigma_12,
    sigma_21,
    sigma_24,
    tau_21,
    tau_32,
):
    return sigma_12 * n1 - sigma_21 * n2 - n2 / tau_21 + n3 / tau_32 - sigma_24 * n2


def dn3_dt(
    n1,
    n3,
    n4,
    na,
    nb,
    Nyb,
    sigma_13,
    sigma_31,
    sigma_35,
    tau_32,
    tau_43,
    R,
):
    return (
        sigma_13 * n1
        - sigma_31 * n3
        - sigma_35 * n3
        + n4 / tau_43
        - n3 / tau_32
        + R * nb * Nyb * n1
        - R * na * Nyb * n3
    )


def dn4_dt(
    n2,
    n4,
    n5,
    sigma_24,
    tau_43,
    tau_54,
):
    return n5 / tau_54 - n4 / tau_43 + sigma_24 * n2


def dn5_dt(
    n3,
    n5,
    sigma_35,
    tau_54,
):
    return sigma_35 * n3 - n5 / tau_54


def dna_dt(
    n1,
    n3,
    na,
    nb,
    Ner,
    sigma_ab,
    sigma_ba,
    tau_ba,
    R,
):
    return (
        -sigma_ab * na
        + sigma_ba * nb
        + nb / tau_ba
        + R * nb * n1 * Ner
        - R * na * n3 * Ner
    )


def dnb_dt(
    n1,
    n3,
    na,
    nb,
    Ner,
    sigma_ab,
    sigma_ba,
    tau_ba,
    R,
):
    return (
        sigma_ab * na
        - sigma_ba * nb
        - nb / tau_ba
        - R * nb * n1 * Ner
        + R * na * n3 * Ner
    )


def _factor_sigma(sigma, nu, P, overlap, A):
    return overlap * sigma * P / (h * nu * A)


def dn1_dt_func(
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    n1,
    n2,
    n3,
    na,
    nb,
    Nyb,
    sigma_a,
    sigma_e,
    sigma_p,
    xi_p,
    tau_21,
    R,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13

    return dn1_dt(
        n1,
        n2,
        n3,
        na,
        nb,
        Nyb,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        tau_21,
        R,
    )


def _dn1_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n1,
    n2,
    n3,
    na,
    nb,
    Nyb,
    sum_a_p_s,
    sum_e_p_s,
    sigma_p,
    xi_p,
    tau_21,
    R,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13

    return dn1_dt(
        n1,
        n2,
        n3,
        na,
        nb,
        Nyb,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        tau_21,
        R,
    )


def dn2_dt_func(
    a_eff,
    overlap_s,
    nu_s,
    P_s,
    n1,
    n2,
    n3,
    sigma_a,
    sigma_e,
    eps_s,
    tau_21,
    tau_32,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)
    sigma_24 = eps_s * sigma_12

    return dn2_dt(
        n1,
        n2,
        n3,
        sigma_12,
        sigma_21,
        sigma_24,
        tau_21,
        tau_32,
    )


def _dn2_dt_func(
    a_eff,
    n1,
    n2,
    n3,
    sum_a_p_s,
    sum_e_p_s,
    eps_s,
    tau_21,
    tau_32,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_24 = eps_s * sigma_12

    return dn2_dt(
        n1,
        n2,
        n3,
        sigma_12,
        sigma_21,
        sigma_24,
        tau_21,
        tau_32,
    )


def dn3_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n1,
    n3,
    n4,
    na,
    nb,
    Nyb,
    sigma_p,
    xi_p,
    eps_p,
    tau_32,
    tau_43,
    R,
):
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_35 = eps_p * sigma_13

    return dn3_dt(
        n1,
        n3,
        n4,
        na,
        nb,
        Nyb,
        sigma_13,
        sigma_31,
        sigma_35,
        tau_32,
        tau_43,
        R,
    )


def dn4_dt_func(
    a_eff,
    overlap_s,
    nu_s,
    P_s,
    n2,
    n4,
    n5,
    sigma_a,
    eps_s,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
    sigma_24 = eps_s * sigma_12

    return dn4_dt(
        n2,
        n4,
        n5,
        sigma_24,
        tau_43,
        tau_54,
    )


def _dn4_dt_func(
    a_eff,
    n2,
    n4,
    n5,
    sum_a_p_s,
    eps_s,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_24 = eps_s * sigma_12

    return dn4_dt(
        n2,
        n4,
        n5,
        sigma_24,
        tau_43,
        tau_54,
    )


def dn5_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n3,
    n5,
    sigma_p,
    eps_p,
    tau_54,
):
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_35 = eps_p * sigma_13

    return dn5_dt(
        n3,
        n5,
        sigma_35,
        tau_54,
    )


def dna_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n1,
    n3,
    na,
    nb,
    Ner,
    sigma_a_p_Y,
    sigma_e_p_Y,
    tau_ba,
    R,
):
    sigma_ab = _factor_sigma(sigma_a_p_Y, nu_p, P_p, overlap_p, a_eff)
    sigma_ba = _factor_sigma(sigma_e_p_Y, nu_p, P_p, overlap_p, a_eff)
    return dna_dt(
        n1,
        n3,
        na,
        nb,
        Ner,
        sigma_ab,
        sigma_ba,
        tau_ba,
        R,
    )


def dnb_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n1,
    n3,
    na,
    nb,
    Ner,
    sigma_a_p_Y,
    sigma_e_p_Y,
    tau_ba,
    R,
):
    sigma_ab = _factor_sigma(sigma_a_p_Y, nu_p, P_p, overlap_p, a_eff)
    sigma_ba = _factor_sigma(sigma_e_p_Y, nu_p, P_p, overlap_p, a_eff)
    return dnb_dt(
        n1,
        n3,
        na,
        nb,
        Ner,
        sigma_ab,
        sigma_ba,
        tau_ba,
        R,
    )


def dPp_dz(
    overlap_p,
    P_p,
    n1,
    n3,
    na,
    nb,
    Ner,
    Nyb,
    sigma_p,
    sigma_a_p_Y,
    sigma_e_p_Y,
    xi_p,
    eps_p,
):
    n1 *= Ner
    n3 *= Ner
    na *= Nyb
    nb *= Nyb
    return (
        (
            -sigma_p * n1
            + sigma_p * xi_p * n3
            - sigma_p * eps_p * n3
            - sigma_a_p_Y * na
            + sigma_e_p_Y * nb
        )
        * overlap_p
        * P_p
    )


def dPs_dz(
    overlap_s,
    P_s,
    n1,
    n2,
    Ner,
    sigma_a,
    sigma_e,
    eps_s,
):
    n1 *= Ner
    n2 *= Ner
    return (-sigma_a * n1 + sigma_e * n2 - sigma_a * eps_s * n2) * overlap_s * P_s


def gain(
    overlap_s,
    n1,
    n2,
    Ner,
    sigma_a,
    sigma_e,
    eps_s,
):
    n1 *= Ner
    n2 *= Ner
    return (-sigma_a * n1 + sigma_e * n2 - sigma_a * eps_s * n2) * overlap_s
