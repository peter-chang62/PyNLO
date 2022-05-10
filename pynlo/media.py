# -*- coding: utf-8 -*-
"""
Optical modes in the frequency domain.

"""

__all__ = ["Mode"]


# %% Imports

import collections

import numpy as np
from scipy.constants import c, pi


# %% Collections

_LinearOperator = collections.namedtuple("LinearOperator", ["u", "phase", "gain", "phase_raw"])


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
    beta_v : array_like of float or callable
        The angular wavenumber, or the real part of the propagation constant,
        defined over the frequency grid.
    alpha_v : array_like of float or callable, optional
        The gain coefficient, or twice the imaginary part of the propagation
        constant. The default is None.
    g2_v : array_like of complex or callable, optional
        The effective 2nd order nonlinearity. The default is None.
    g2_inv : callable, optional
        Discrete polling of the 2nd order nonlinearity. The default is None.
    g3_v : array_like of complex or callable, optional
        The effective 3rd order nonlinearity. The default is None.
    rv_grid : array_like of float, optional
        An origin contiguous frequency grid associated with the 3rd order
        nonlinear response function. The default is None.
    r3_v : array_like of complex or callable, optional
        The effective 3rd order nonlinear response function containing both
        the instantaneous and Raman nonlinearities. The default is None.
    z : float, optional
        The position along the waveguide. The default is 0.0.

    Notes
    -----
    Modes are defined for traveling waves of the form assumed below:

    .. math:: E, H \\sim a \\, e^{i(\\omega t - \\kappa z)} + \\text{c.c} \\\\
              \\kappa = \\beta + i \\frac{\\alpha}{2}, \\quad
              \\beta = n \\frac{\\omega}{c}

    """

    def __init__(self, v_grid, beta_v, alpha_v=None,
                 g2_v=None, g2_inv=None, g3_v=None, rv_grid=None, r3_v=None,
                 z=0.0):
        """
        Initialize a mode given a set of frequencies, wavenumbers, and other
        parameters. If a given parameters is callable, its first argument must
        be `z`, the position along the waveguide.

        Parameters
        ----------
        v_grid : array_like of float
            The frequency grid.
        beta_v : array_like of float or callable
            The angular wavenumber, or the real part of the propagation
            constant, defined over the frequency grid.
        alpha_v : array_like of float or callable, optional
            The gain coefficient, or twice the imaginary part of the
            propagation constant. The default is None.
        g2_v : array_like of complex or callable, optional
            The effective 2nd order nonlinearity. The default is None.
        g2_inv : callable, optional
            Discrete polling of the 2nd order nonlinearity. The default is
            None.
        g3_v : array_like of complex or callable, optional
            The effective 3rd order nonlinearity. The default is None.
            An origin contiguous frequency grid associated with the 3rd order
            nonlinear response function. The default is None.
        r3_v : array_like of complex or callable, optional
            The effective 3rd order nonlinear response function containing
            both the instantaneous and Raman nonlinearities. The default is
            None.
        z : float, optional
            The position along the waveguide. The default is 0.

        """
        #---- Position
        self._z = z

        #---- Frequency Grid
        self._v_grid = np.asarray(v_grid, dtype=float)
        self._w_grid = 2*pi*self._v_grid

        #---- Refractive Index
        if callable(beta_v):
            assert (len(beta_v(z)) == len(v_grid)), "The length of beta_v must match v_grid."
            self._beta = beta_v
        else:
            assert (len(beta_v) == len(v_grid)), "The length of beta_v must match v_grid."
            self._beta = np.asarray(beta_v, dtype=float)

        #---- Gain
        if (alpha_v is None) or callable(alpha_v):
            self._alpha = alpha_v
        else:
            self._alpha = np.asarray(alpha_v, dtype=float)

        #---- 2nd Order Nonlinearity
        if (g2_v is None) or callable(g2_v):
            self._g2 = g2_v
        else:
            self._g2 = np.asarray(g2_v, dtype=complex)

        if (g2_inv is None) or callable(g2_inv):
            self._g2_inv = g2_inv
        else:
            assert callable(g2_inv), ("If defined, the discrete 2nd order"
                                       "polling must be callable")

        #---- 3rd Order Nonlinearity
        if (g3_v is None) or callable(g3_v):
            self._g3 = g3_v
        else:
            self._g3 = np.asarray(g3_v, dtype=complex)

        if (rv_grid is not None) and (r3_v is not None):
            self._rv_grid = np.asarray(rv_grid, dtype=float)

            #---- Nonlinear Response Function
            if callable(r3_v):
                assert (len(r3_v(z)) == len(rv_grid)), "The length of r3_v must match rv_grid."
                self._r3 = r3_v
            else:
                assert (len(r3_v) == len(rv_grid)), "The length of r3_v must match rv_grid."
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

    @property
    def z_dep_linearity(self):
        """
        The z dependence of the linear terms.

        Returns
        -------
        z_alpha : bool
            The z dependence of gain constant.
        z_beta : bool
            The z dependence of angular wavenumber.

        """
        z_alpha = callable(self._alpha)
        z_beta = callable(self._beta)
        return z_alpha, z_beta

    @property
    def z_dep_nonlinearity(self):
        """
        The z dependence of the nonlinear terms.

        Returns
        -------
        z_g2 : bool
            The z dependence of the effective 2nd order nonlinear parameter.
        z_g2_inv : bool
            The z dependence of the polling of the effective 2nd order
            nonlinear parameter.
        z_g3 : bool
            The z dependence of the effective 3rd order nonlinear parameter.
        z_r3 : bool
            The z dependence of the effective Raman response.

        """
        z_g2 = callable(self._g2)
        z_g2_inv = callable(self._g2_inv)
        z_g3 = callable(self._g3)
        z_r3 = callable(self._r3)
        return z_g2, z_g2_inv, z_g3, z_r3

    #---- 1st Order Properties
    def beta(self, m=0, z=None):
        """
        The angular wavenumber, with units of ``1/m``, or its derivatives with
        respect to angular frequency.

        This method is recursive and succesively calculates higher order
        derivatives from the lower orders. This method returns the refractive
        index unchanged when `m` is less than or equal to 0.

        Parameters
        ----------
        m : int, optional
            The derivative order of the propagation constant with respect to
            angular frequency. The default returns the propagation constant
            without taking a derivative.
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        ndarray of float

        """
        if z is not None:
            self.z = z

        if m<=0:
            return self._beta(self.z) if callable(self._beta) else self._beta
        else:
            return np.gradient(self.beta(m=m-1), self._w_grid, edge_order=2)

    def n(self, z=None):
        """
        The refractive index.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        ndarray of float

        """
        if z is not None:
            self.z = z
        return self.beta()*c/self._w_grid

    def alpha(self, z=None):
        """
        The gain constant, with units of ``1/m``.

        Positive values correspond to gain and negative values to loss.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        None or ndarray of float

        """
        if z is not None:
            self.z = z
        return self._alpha(self.z) if callable(self._alpha) else self._alpha

    def n_g(self, z=None):
        """
        The group index.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

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
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        ndarray of float

        """
        if z is not None:
            self.z = z
        return 1/self.beta(m=1)

    def d_12(self, v0=None, z=0):
        """
        The group velocity mismatch, or walk-off parameter, with units of
        ``s/m``.

        Parameters
        ----------
        v0 : float, optional
            The target reference frequency. The default selects the central
            frequency.
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        ndarray of float

        """
        if z is not None:
            self.z = z

        if v0 is None: # comoving frame
            v0 = self.v_grid[self.v_grid.size//2]
        v0_idx = np.argmin(np.abs(v0 - self.v_grid))
        beta1 = self.beta(m=1)
        return beta1[v0_idx] - beta1

    def GVD(self, z=None):
        """
        The group velocity dispersion, with units of ``s**2/m``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

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
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        ndarray of float

        """
        if z is not None:
            self.z = z
        return -2*pi/c * self.v_grid**2 * self.beta(m=2)

    def linear_operator(self, dz, v0=None, z=None):
        """
        The linear operator which advances the spectrum over a distance `dz`.

        The linear operator acts on the spectrum through multiplication.

        Parameters
        ----------
        dz : float
            The step size.
        v0 : float, optional
            The target reference frequency of the comoving frame. The default
            selects the central frequency.
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

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

        #---- Gain
        alpha = self.alpha()
        if alpha is None:
            alpha = 0.0
        gain = np.exp(alpha*dz)

        #---- Phase
        beta_raw = self.beta()

        if v0 is None: # comoving frame
            v0 = self.v_grid[self.v_grid.size//2]
        v0_idx = np.argmin(np.abs(v0 - self.v_grid))
        beta_cm = beta_raw - self.beta(m=1)[v0_idx]*self._w_grid

        #---- Propagation Constant
        kappa = beta_cm + 0.5j*alpha

        #---- Linear Operator
        operator = np.exp(-1j*kappa*dz)

        lin_operator = _LinearOperator(
            u=operator, phase=dz*beta_cm, gain=gain, phase_raw=dz*beta_raw)
        return lin_operator

    #---- 2nd Order Properties
    def g2(self, z=None):
        """
        The magnitude of the effective 2nd order nonlinear parameter, with
        units of ``1/(W**0.5*m*Hz)``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        None or ndarray of complex

        """
        if z is not None:
            self.z = z
        return self._g2(self.z) if callable(self._g2) else self._g2

    def g2_inv(self, z=None):
        """
        The sign of the polled 2nd order nonlinearity.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        int

        """
        if z is not None:
            self.z = z
        return 1 if self._g2_inv is None else self._g2_inv(self.z)

    #---- 3rd Order Properties
    def g3(self, z=None):
        """
        The effective 3rd order nonlinear parameter, with units of
        ``1/(W*m*Hz)``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        None or ndarray of complex

        """
        if z is not None:
            self.z = z
        return self._g3(self.z) if callable(self._g3) else self._g3

    def gamma(self, z=None):
        """
        The nonlinear parameter :math:`\\gamma`, with units of ``1/(W*m)``.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        None or ndarray of complex
        """
        if z is not None:
            self.z = z

        g3 = self.g3()
        if g3 is not None and len(g3.shape)>=2:
            g3 = g3[0] * np.sum(g3[1:]**3, axis=0)
        return 3/2*self._w_grid*g3 if g3 is not None else None


    def r3(self, z=None):
        """
        The effective 3rd order nonlinear response function containing both
        the instantaneous and Raman nonlinearities.

        Parameters
        ----------
        z : float, optional
            The position along the waveguide. The default uses the last known
            value.

        Returns
        -------
        None or ndarray of complex
        """
        if z is not None:
            self.z = z
        return self._r3(self.z) if callable(self._r3) else self._r3


# class Waveguide():
#     """
#     Collection of modes and the nonlinear interactions between them
#     """
#     def __init__(self, modes, coupling):
#         pass

# class GaussianMode(Waveguide):
#     """
#     Collection of Hermite–Gaussian or Laguerre–Gaussian modes for simulating
#     free space propagation
#     - effective area based on distance to nominal waist location
#     - could also include convenience functions for setting up the beam
#       through focusing, propagation, etc.
#
#     """
#     def __init__(self, modes, coupling):
#         pass
