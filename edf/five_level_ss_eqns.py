"""
this scratch file is to re-create just the rate equations (not PyNLO) but this
time including pump excited state absorption (ESA).
"""

# %% ----- imports
import numpy as np
from scipy.constants import h

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# pump emission and esa cross-sections
xi_p = 1.08  # sigma_31 / sigma_13: ratio of sigma_e to sigma_a at 980 nm
eps_p = 0.95  # sigma_35 / sigma_13: ratio of sigma_esa to sigma_a at 980 nm
eps_s = 0.17  # sigma_24/sigma_12: ratio of sigma_esa to sigma_a for the signal

# lifetimes
tau_21 = 10 * ms
tau_32 = 5.2 * us
tau_43 = 5 * ns
tau_54 = 1 * us

# -----------------------------------------------------------------------------
# solving for the 5 level system with levels coupled the same way as:
#
#   Barmenkov et al. Journal of Applied Physics 106, 083108 (2009).
#
# pulling n1, n2, and n3 calculated in mathematica and ported to python using
# parse_mathematica from sympy. In all the expressions below:
#
#   sigma = overlap * sigma * P / (h * nu * A)
# -----------------------------------------------------------------------------


def _n1_mathematica(
    n,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    sigma_24,
    sigma_35,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    return (
        n
        * (
            sigma_21 * (sigma_31 * tau_21 * tau_32 + tau_21)
            + sigma_31 * (sigma_24 * tau_21 * tau_32 + tau_32)
            + 1
        )
        / (
            sigma_12
            * tau_21
            * (
                sigma_24
                * (
                    tau_32
                    * (sigma_31 * tau_43 + sigma_35 * tau_43 + sigma_35 * tau_54 + 1)
                    + tau_43
                )
                + sigma_31 * tau_32
                + 1
            )
            + sigma_13
            * (
                sigma_21 * tau_21 * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32)
                + sigma_24
                * tau_21
                * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32 + tau_43)
                + sigma_35 * tau_32 * (tau_43 + tau_54)
                + tau_21
                + tau_32
            )
            + sigma_21 * sigma_31 * tau_21 * tau_32
            + sigma_21 * tau_21
            + sigma_24 * sigma_31 * tau_21 * tau_32
            + sigma_31 * tau_32
            + 1
        )
    )


def _n2_mathematica(
    n,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    sigma_24,
    sigma_35,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    return (
        tau_21
        * n
        * (sigma_12 * sigma_31 * tau_32 + sigma_12 + sigma_13)
        / (
            sigma_12
            * tau_21
            * (
                sigma_24
                * (
                    tau_32
                    * (sigma_31 * tau_43 + sigma_35 * tau_43 + sigma_35 * tau_54 + 1)
                    + tau_43
                )
                + sigma_31 * tau_32
                + 1
            )
            + sigma_13
            * (
                sigma_21 * tau_21 * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32)
                + sigma_24
                * tau_21
                * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32 + tau_43)
                + sigma_35 * tau_32 * (tau_43 + tau_54)
                + tau_21
                + tau_32
            )
            + sigma_21 * sigma_31 * tau_21 * tau_32
            + sigma_21 * tau_21
            + sigma_24 * sigma_31 * tau_21 * tau_32
            + sigma_31 * tau_32
            + 1
        )
    )


def _n3_mathematica(
    n,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    sigma_24,
    sigma_35,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    return (
        tau_32
        * n
        * (
            sigma_12 * sigma_24 * tau_21
            + sigma_13 * sigma_21 * tau_21
            + sigma_13 * sigma_24 * tau_21
            + sigma_13
        )
        / (
            sigma_12
            * tau_21
            * (
                sigma_24
                * (
                    tau_32
                    * (sigma_31 * tau_43 + sigma_35 * tau_43 + sigma_35 * tau_54 + 1)
                    + tau_43
                )
                + sigma_31 * tau_32
                + 1
            )
            + sigma_13
            * (
                sigma_21 * tau_21 * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32)
                + sigma_24
                * tau_21
                * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32 + tau_43)
                + sigma_35 * tau_32 * (tau_43 + tau_54)
                + tau_21
                + tau_32
            )
            + sigma_21 * sigma_31 * tau_21 * tau_32
            + sigma_21 * tau_21
            + sigma_24 * sigma_31 * tau_21 * tau_32
            + sigma_31 * tau_32
            + 1
        )
    )


def _n4_mathematica(
    n,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    sigma_24,
    sigma_35,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    return (
        tau_43
        * n
        * (
            sigma_12 * sigma_24 * tau_21 * (sigma_31 * tau_32 + sigma_35 * tau_32 + 1)
            + sigma_13 * sigma_24 * (sigma_35 * tau_21 * tau_32 + tau_21)
            + sigma_13 * sigma_35 * tau_32 * (sigma_21 * tau_21 + 1)
        )
        / (
            sigma_12
            * tau_21
            * (
                sigma_24
                * (
                    tau_32
                    * (sigma_31 * tau_43 + sigma_35 * tau_43 + sigma_35 * tau_54 + 1)
                    + tau_43
                )
                + sigma_31 * tau_32
                + 1
            )
            + sigma_13
            * (
                sigma_21 * tau_21 * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32)
                + sigma_24
                * tau_21
                * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32 + tau_43)
                + sigma_35 * tau_32 * (tau_43 + tau_54)
                + tau_21
                + tau_32
            )
            + sigma_21 * sigma_31 * tau_21 * tau_32
            + sigma_21 * tau_21
            + sigma_24 * sigma_31 * tau_21 * tau_32
            + sigma_31 * tau_32
            + 1
        )
    )


def _n5_mathematica(
    n,
    sigma_12,
    sigma_21,
    sigma_13,
    sigma_31,
    sigma_24,
    sigma_35,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    return (
        sigma_35
        * tau_32
        * tau_54
        * n
        * (
            sigma_12 * sigma_24 * tau_21
            + sigma_13 * sigma_21 * tau_21
            + sigma_13 * sigma_24 * tau_21
            + sigma_13
        )
        / (
            sigma_12
            * tau_21
            * (
                sigma_24
                * (
                    tau_32
                    * (sigma_31 * tau_43 + sigma_35 * tau_43 + sigma_35 * tau_54 + 1)
                    + tau_43
                )
                + sigma_31 * tau_32
                + 1
            )
            + sigma_13
            * (
                sigma_21 * tau_21 * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32)
                + sigma_24
                * tau_21
                * (sigma_35 * tau_32 * (tau_43 + tau_54) + tau_32 + tau_43)
                + sigma_35 * tau_32 * (tau_43 + tau_54)
                + tau_21
                + tau_32
            )
            + sigma_21 * sigma_31 * tau_21 * tau_32
            + sigma_21 * tau_21
            + sigma_24 * sigma_31 * tau_21 * tau_32
            + sigma_31 * tau_32
            + 1
        )
    )


def _factor_sigma(sigma, nu, P, overlap, A):
    return overlap * sigma * P / (h * nu * A)


def n1_func(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n1_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def _n1_func(
    n,
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    sigma_p,
    sum_a_p_s,
    sum_e_p_s,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n1_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def n2_func(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n2_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def _n2_func(
    n,
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    sigma_p,
    sum_a_p_s,
    sum_e_p_s,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n2_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def n3_func(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n3_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def _n3_func(
    n,
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    sigma_p,
    sum_a_p_s,
    sum_e_p_s,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n3_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def n4_func(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n4_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def _n4_func(
    n,
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    sigma_p,
    sum_a_p_s,
    sum_e_p_s,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n4_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def n5_func(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n5_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def _n5_func(
    n,
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    sigma_p,
    sum_a_p_s,
    sum_e_p_s,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    sigma_12 = sum_a_p_s
    sigma_21 = sum_e_p_s
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)
    sigma_31 = xi_p * sigma_13
    sigma_24 = eps_s * sigma_12
    sigma_35 = eps_p * sigma_13

    return _n5_mathematica(
        n,
        sigma_12,
        sigma_21,
        sigma_13,
        sigma_31,
        sigma_24,
        sigma_35,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    )


def dPp_dz(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    args = [
        n,
        a_eff,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p,
        sigma_a,
        sigma_e,
        eps_p,
        xi_p,
        eps_s,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    ]
    n1 = n1_func(*args)
    n3 = n3_func(*args)

    return (
        (-sigma_p * n1 + sigma_p * xi_p * n3 - sigma_p * eps_p * n3) * overlap_p * P_p
    )


def dPs_dz(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    args = [
        n,
        a_eff,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p,
        sigma_a,
        sigma_e,
        eps_p,
        xi_p,
        eps_s,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    ]
    n1 = n1_func(*args)
    n2 = n2_func(*args)

    return (-sigma_a * n1 + sigma_e * n2 - sigma_a * eps_s * n2) * overlap_s * P_s


# same as dPs_dz but without the multiplying factor of P_s
def gain(
    n,
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p,
    sigma_a,
    sigma_e,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
):
    args = [
        n,
        a_eff,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p,
        sigma_a,
        sigma_e,
        eps_p,
        xi_p,
        eps_s,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    ]
    n1 = n1_func(*args)
    n2 = n2_func(*args)

    return (-sigma_a * n1 + sigma_e * n2 - sigma_a * eps_s * n2) * overlap_s
