# -*- coding: utf-8 -*-
"""
Module containing conversion functions and other calculators relevant to the
2nd order nonlinear susceptibility.

"""

__all__ = []


# %% Imports

import numpy as np
from scipy.constants import pi, c, epsilon_0 as e0


# %% Converters

#---- 2rd Order Nonlinear Susceptibilities chi2 and d
def d_to_chi2(d):
    return d * 2

def chi2_to_d(chi2):
    return chi2 / 2

#---- Polling Period and Wavenumber Mismatch
def dk_to_period(dk):
    return 2*pi/dk

def period_to_dk(period):
    return 2*pi/period


# %% Estimators

#TODO: setup media, solver to use slip g's

def g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff):
    # for cascaded chi2
    # v0 is pump frequency
    # (2*v0 < v_grid).any()
    pass #TODO

def g2_split(n_eff, a_eff, chi2_eff):
    return np.array([1/2 * ((e0*a_eff)/(c*n_eff))**0.5 * chi2_eff,
                     1/(e0*c*n_eff*a_eff)**0.5]).T

def g2_least_phase(n_eff, a_eff, chi2_eff, paths):
    g2s = g2_split(n_eff, a_eff, chi2_eff)
    assert g2s.size != len(paths), "The number of paths must equal the number of frequencies."

    g2_p = []
    for idx, path in enumerate(paths):
        if None in path:
            g2_p.append(0.0)
        else:
            g2 = np.mean([g2s[idx, 0] * g2s[cpl[0], 1]*g2s[cpl[0], 1] for cpl in path])
            g2_p.append(g2)
    return np.arary(g2_p)

def effective_chi3():
    pass #TODO: effective 3rd order from cascaded 2nd


# %% Helper Functions

def poll_sign(z, z_invs):
    r"""
    Calculate the sign of a discrete quasi-phase matching (QPM) structure
    given the locations of each inversion.

    Parameters
    ----------
    z : float
        The position along the waveguide.
    z_invs : array_like of float
        The locations along the waveguide where the sign of the polling
        changes.

    Returns
    -------
    int

    Notes
    -----
    For continous QPM profiles, the inversion locations can
    be calculated by inverting the integral equation that gives the total
    number of periods:

    .. math::
        \text{cycles}[z] &= \int_{z_0}^z \frac{\Delta k[z^\prime]}{2 \pi} dz^\prime
                          = \int_{z_0}^z \frac{dz^\prime}{\text{period}[z^\prime]} \\
        \text{z}_{inv}[n] &= \text{cycles}^{-1}[n/2]

    where ":math:`\Delta k`" is the wavenumber mismatch compensated by the
    polling and ":math:`n`" is an integer.

    """
    return 1-2*(np.count_nonzero(z_invs <= z) % 2)

def dominant_paths(v_grid, beta, beta_qpm=None, full=False):
    """
    For each output frequency, find the input frequencies that are coupled
    with the least phase mismatch.

    This function checks both sum frequency generation (SFG) pathways and
    difference frequency generation (DFG) pathways.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    beta : array_like of float
        The wavenumber of each input frequency.
    beta_qpm : float, optional
        The effective wavenumber of an applied quasi-phase matching (QPM)
        structure. The default output is calculated without a QPM structure.
    full : bool, optional
        Switch determining nature of the return value. When it is ``False``
        (the default) just the path indices are returned, when ``True`` the
        calculated wavenumber mismatch arrays are also returned.

    Returns
    -------
    paths : list
        Pairs of indices for each output frequency that correspond to the
        input frequencies of the dominant input paths. If no valid path exists
        for a particular output frequency its input indices are given as
        ``[None, None]``.
    (dk, dk_sfg, dk_dfg) : tuple
        These values are only returned if ``full=True`` \n
        dk : list
            The wavenumber mismatch for each path in `paths`.
        v_sfg : ndarray of float
            The frequencies that correspond to all SFG combinations.
        dk_sfg : ndarray of float
            The wavenumber mismatch for all SFG combinations. The mismatch of
            invalid paths are given as NaN.
        v_dfg : ndarray of float
            The frequencies that correspond to all DFG combinations.
        dk_dfg : ndarray of float
            The wavenumber mismatch for all DFG combinations. The mismatch of
            invalid paths are given as NaN.

    """
    #---- Setup
    v_grid = np.asarray(v_grid, dtype=float)
    n = v_grid.size
    v_idx = np.arange(n)
    v_idx2 = np.dstack(np.indices((n,n)))
    beta = np.asarray(beta, dtype=float)

    #---- Sum Frequency
    v_sfg = np.add.outer(v_grid, v_grid)

    # Indexing
    v_idx_sfg = np.interp(v_sfg, v_grid, v_idx, left=np.nan, right=np.nan).round()
    sfg_idxs = np.arange(np.nanmin(v_idx_sfg), np.nanmax(v_idx_sfg)+1, dtype=int)
    sfg_diag_offset = (n-1) - np.arange(sfg_idxs.size)

    # Wavenumber Mismatch
    beta_sfg_12 = np.add.outer(beta, beta)
    beta_sfg_3 = np.interp(v_sfg, v_grid, beta, left=np.nan, right=np.nan)
    if beta_qpm is None:
        dk_sfg = np.abs(beta_sfg_12 - beta_sfg_3)
    else:
        dk_sfg = np.abs(np.abs(beta_sfg_12 - beta_sfg_3) - beta_qpm)

    #---- Difference Frequency
    v_dfg = np.subtract.outer(v_grid, v_grid)

    # Indexing
    v_idx_dfg = np.interp(v_dfg, v_grid, v_idx, left=np.nan, right=np.nan).round()
    dfg_idxs = np.arange(np.nanmin(v_idx_dfg), np.nanmax(v_idx_dfg)+1, dtype=int)
    dfg_diag_offset = -(n-1) + np.arange(dfg_idxs.size)[::-1]

    # Wavenumber Mismatch
    beta_dfg_31 = np.subtract.outer(beta, beta)
    beta_dfg_2 = np.interp(v_dfg, v_grid, beta, left=np.nan, right=np.nan)
    if beta_qpm is None:
        dk_dfg = np.abs(beta_dfg_31 - beta_dfg_2)
    else:
        dk_dfg = np.abs(np.abs(beta_dfg_31 - beta_dfg_2) - beta_qpm)

    #---- Paths of Minimum Phase Mismatch
    paths = []
    dk = []
    for idx in v_idx:
        valid_sfg = (idx in sfg_idxs)
        valid_dfg = (idx in dfg_idxs)

        # SFG
        if valid_sfg:
            diag_offset = sfg_diag_offset[idx==sfg_idxs][0]
            sfg_diag = np.fliplr(dk_sfg).diagonal(diag_offset)
            min_dk_sfg = np.nanmin(sfg_diag)
            min_dk_sfg_idx = np.nonzero(sfg_diag==min_dk_sfg)
            sfg_path = np.fliplr(v_idx2).diagonal(diag_offset).T[min_dk_sfg_idx]

        # DFG
        if valid_dfg:
            diag_offset = dfg_diag_offset[idx==dfg_idxs][0]
            dfg_diag = dk_dfg.diagonal(diag_offset)
            min_dk_dfg = np.nanmin(dfg_diag)
            min_dk_dfg_idx = np.nonzero(dfg_diag==min_dk_dfg)
            dfg_path = v_idx2.diagonal(diag_offset).T[min_dk_dfg_idx]

        if valid_sfg and valid_dfg:
            if min_dk_sfg < min_dk_dfg:
                # SFG smaller
                paths.append(sfg_path.tolist())
            elif min_dk_sfg > min_dk_dfg:
                # DFG smaller
                paths.append(dfg_path.tolist())
            else:
                # Equal
                paths.append(sfg_path.tolist() + dfg_path.tolist())
            dk.append(min_dk_sfg) if min_dk_sfg < min_dk_dfg else dk.append(min_dk_dfg)
        elif valid_sfg:
            # Only SFG
            paths.append(sfg_path.tolist())
            dk.append(min_dk_sfg)
        elif valid_dfg:
            # Only DFG
            paths.append(dfg_path.tolist())
            dk.append(min_dk_dfg)
        else:
            # No path
            paths.append([[None, None]])
            dk.append(None)

    if full:
        return paths, (dk, v_sfg, dk_sfg, v_dfg, dk_dfg)
    else:
        return paths
