# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = ["Mode"]


# %% Imports

import collections

import numpy as np
from scipy import constants

from pynlo.utility import fft


# %% Constants

pi = constants.pi
c = constants.c
e0 = constants.epsilon_0


# %% Collections

LinearOperator = collections.namedtuple("LinearOperator", ["u", "phase", "gain", "phase_raw"])


# %% Base Classes
class Mode():
    """
    An optical mode, defined over a frequency grid.

    A given input parameter is interpreted as being z-dependent if it is
    callable. All z-dependent parameters must have the position along the
    waveguide (`z`) as their first argument.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    n_v : array_like of float or callable
        The refractive indicies defined over the frequency grid.
    alpha_v : array_like of float or callable, optional
        The gain coefficient. The default is None.
    g2_v : array_like of complex or callable, optional
        The effective 2nd order nonlinearity. The default is None.
    g3_v : array_like of complex or callable, optional
        The effective 3rd order nonlinearity. The default is None.
    rv_grid : array_like of float, optional
        An origin contiguous frequency grid associated with the Raman
        response. The default is None.
    r3_v : array_like of complex or callable, optional
        The efective Raman response. The default is None.
    z : float, optional
        The position along the waveguide. The default is 0.0.

    Notes
    -----
    TODO: add notes
    """
    def __init__(self, v_grid, n_v, alpha_v=None,
                 g2_v=None, g3_v=None, rv_grid=None, r3_v=None, z=0.0):
        """
        Initialize a mode given a set of frequencies, refractive indices, and
        other parameters. If a given parameters is callable, its first
        argument must be `z`, the position along the waveguide.

        Parameters
        ----------
        v_grid : array_like of float
            The frequency grid.
        n_v : array_like of float or callable
            The refractive indicies defined over the frequency grid.
        alpha_v : array_like of float or callable, optional
            The gain coefficient. The default is None.
        g2_v : array_like of complex or callable, optional
            The effective 2nd order nonlinearity. The default is None.
        g3_v : array_like of complex or callable, optional
            The effective 3rd order nonlinearity. The default is None.
        rv_grid : array_like of float, optional
            An origin contiguous frequency grid associated with the Raman
            response. The default is None.
        r3_v : array_like of complex or callable, optional
            The efective Raman response. The default is None.
        z : float, optional
            The position along the waveguide. The default is 0.0.
        """
        #---- Position
        self._z = z

        #---- Frequency Grid
        self._v_grid = np.asarray(v_grid, dtype=float)

        #---- Refractive Index
        if callable(n_v):
            assert (len(n_v(z)) == len(v_grid)), "The length of n_v must match v_grid."
            self._n = n_v
        else:
            assert (len(n_v) == len(v_grid)), "The length of n_v must match v_grid."
            self._n = np.asarray(n_v, dtype=float)

        #---- Loss
        if (alpha_v is None) or callable(alpha_v):
            self._alpha = alpha_v
        else:
            self._alpha = np.asarray(alpha_v, dtype=float)

        #---- 2nd Order Nonlinearity
        if (g2_v is None) or callable(g2_v):
            self._g2 = g2_v
        else:
            self._g2 = np.asarray(g2_v, dtype=complex)

        #---- 3rd Order Nonlinearity
        if (g3_v is None) or callable(g3_v):
            self._g3 = g3_v
        else:
            self._g3 = np.asarray(g3_v, dtype=complex)

        if (rv_grid is not None) and (r3_v is not None):
            self._rv_grid = np.asarray(rv_grid, dtype=float)

            #---- Raman Nonlinearity
            if callable(r3_v):
                self._r3 = r3_v
            else:
                self._r3 = np.asarray(r3_v, dtype=complex)
        else:
            assert (rv_grid is not None)==(r3_v is not None), (
                "Both rv_grid and r3_v must be defined at the same time.")
            self._rv_grid = None
            self._r3 = None

    #---- General Properties
    @property
    def z(self):
        """
        The position along the waveguide, with units of ``m``.

        Returns
        -------
        float
        """
        return self._z
    @z.setter
    def z(self, z):
        self._z = z

    @property
    def v_grid(self):
        """
        The frequency grid, with units of ``Hz``.

        Returns
        -------
        ndarray of float
        """
        return self._v_grid

    @property
    def rv_grid(self):
        """
        The origin contiguous frequency grid associated with the Raman
        response. Units are in ``Hz``.

        Returns
        -------
        None or ndarray of float
        """
        return self._rv_grid

    #---- 1st Order Properties
    def n(self, z=None):
        """
        The refractive index.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        if callable(self._n):
            return self._n(self.z)
        else:
            return self._n

    def dndv(self, m=1, z=None):
        """
        The derivative of the refractive index with repsect to frequency.

        This method is recursive and succesively calculates higher order
        derivatives from the lower orders. This method returns the refractive
        index unchanged when `m` is less than or equal to 0.

        Parameters
        ----------
        m : int, optional
            The order of the returned derivative. The default is 1.
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        assert isinstance(m, (int, np.integer)), "The derivative order must be an integer."

        if z is not None:
            self.z = z

        #--- Calculate Derivatives
        if m <= 0:
            dndv = self.n()
        else:
            dndv = np.gradient(self.dndv(m=m-1), self.v_grid)
        return dndv

    def alpha(self, z=None):
        """
        The gain constant, with units of ``1/m``.

        Positive values correspond to gain and negative values to loss.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        None or ndarray of float
        """
        if z is not None:
            self.z = z

        if callable(self._alpha):
            return self._alpha(self.z)
        else:
            return self._alpha

    def beta(self, m=0, z=None):
        """
        The propagation constant or angular wavenumber, with units of ``1/m``.

        Parameters
        ----------
        m : int, optional
            The order of the returned derivative of the propagation constant.
            The default is 0, which returns the propagation constant without
            taking the derivative.
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        return 1/c * (2*pi)**(1-m) * (m*self.dndv(m=m-1) + self.v_grid*self.dndv(m=m))

    def n_g(self, z=None):
        """
        The group index.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        return c*self.beta(m=1)

    def v_g(self, z=None):
        """
        The group velocity, with units of ``m/s``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        return 1/self.beta(m=1)

    def GVD(self, z=None):
        """
        The group velocity dispersion, with units of ``s**2/m``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        return self.beta(m=2)

    def D(self, z=None):
        """
        The dispersion parameter D, with units of ``s/m**2``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        ndarray of float
        """
        if z is not None:
            self.z = z

        return -2*pi/c * self.v_grid**2 * self.beta(m=2)

    def linear_operator(self, dz, v_0=None, z=None):
        """
        The linear operator which advances the spectrum over a distance `dz`.

        The linear operator acts on the spectrum through multiplication.

        Parameters
        ----------
        dz : float
            The step size.
        v_0 : float, optional
            The target reference frequency that sets the position of the
            comoving frame. The default is None, which chooses the central
            frequency.
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        u : ndarray of complex
            The forward evolution operator.
        phase : ndarray of float
            The accumulated phase in the comoving frame (additive).
        gain : float or ndarray
            The accumulated gain or loss (multiplicative).
        phase_raw : ndarray of float
            The raw accumulated phase.
        """
        if z is not None:
            self.z = z

        #--- Phase and Loss
        phase_raw = dz*self.beta()
        gain = 0.5j*dz*self.alpha() if self.alpha() is not None else 0.0

        #--- Comoving Reference Frame
        if v_0 is None:
            v_0 = fft.ifftshift(self.v_grid)[0]
        v_0_idx = np.argmin(np.abs(v_0 - self.v_grid))
        phase_cm = 2*pi*dz*self.beta(m=1)[v_0_idx]*self.v_grid
        phase = phase_raw - phase_cm

        #--- Linear Operator
        operator = np.exp(1j*(phase + gain))

        lin_operator = LinearOperator(
            u=operator, phase=phase, gain=np.exp(gain), phase_raw=phase_raw)
        return lin_operator

    #---- 2nd Order Properties
    def g2(self, z=None):
        """
        The effective 2nd order nonlinear parameter, with units of
        ``1/(W**0.5*m*Hz)``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        None or ndarray of complex
        """
        if z is not None:
            self.z = z

        if callable(self._g2):
            return self._g2(self.z)
        else:
            return self._g2

    #---- 3rd Order Properties
    def g3(self, z=None):
        """
        The effective 3rd order nonlinear parameter, with units of
        ``1/(W*m*Hz).

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        None or ndarray of complex
        """
        if z is not None:
            self.z = z

        if callable(self._g3):
            return self._g3(self.z)
        else:
            return self._g3

    def gamma(self, z=None):
        r"""
        The nonlinear parameter :math:`\gamma`, with units of ``1/(W*m)``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        None or ndarray of float
        """
        if z is not None:
            self.z = z

        if self.g3() is not None:
            return (3/2 * 2*pi*self.v_grid * self.g3()).real
        else:
            return None

    def r3(self, z=None):
        """
        The effective raman response, with units of ``1/Hz``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default is to use the last
            known value.

        Returns
        -------
        None or ndarray of complex
        """
        if z is not None:
            self.z = z

        if callable(self._r3):
            return self._r3(self.z)
        else:
            return self._r3


# class Waveguide():
#     """
#     Collection of modes and the nonlinear interactions between them
#     """
#     def __init__(self, modes, coupling):
#         pass
