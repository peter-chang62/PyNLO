# -*- coding: utf-8 -*-
"""
Time and frequency grid utilites and and other miscellaneous helper functions.

The submodules contain calculator type functions for converting between
physically relevant parameters related to the linear and nonlinear
susceptibilities as well as an efficient interface to fast fourier transforms.

"""

__all__ = ["chi1", "chi2", "chi3", "fft",
           "taylor_series",
           "resample_v", "resample_t", "derivative_v", "derivative_t",
           "TFGrid"]


# %% Imports

import collections

import numpy as np
from scipy.constants import pi, h

from pynlo.utility import chi1, chi2, chi3, fft


# %% Collections

_ResampledV = collections.namedtuple("ResampledV", ["v_grid", "f_v", "dv", "dt"])

_ResampledT = collections.namedtuple("ResampledT", ["t_grid", "f_t", "dt"])

_RTFGrid = collections.namedtuple("RTFGrid", ["n", "v0",
                                             "v_grid", "v_ref", "dv", "v_window",
                                             "t_grid", "t_ref", "dt", "t_window"])


# %% Routines

def taylor_series(x0, derivatives):
    """
    Calculate a Taylor series expansion given the derivatives of a function
    about a point.

    Parameters
    ----------
    x0 : float
        The center point of the Taylor series expansion.
    derivatives : array_like
        The derivatives of the function with respect to `x` evaluated at `x0`.
        The coefficients must be given in order of increasing degree, i.e.
        ``[f(x0), f'(x0), f''(x0), ...]``.

    Returns
    -------
    pwr_series : numpy.polynomial.Polynomial
        A NumPy `Polynomial` object representing the Taylor series expansion.

    """
    window = np.array([-1, 1])
    domain = window + x0
    poly_coefs = [coef/np.math.factorial(n) for (n, coef) in enumerate(derivatives)]
    pwr_series = np.polynomial.Polynomial(poly_coefs, domain=domain, window=window)
    return pwr_series

def coherent_noise_v(v_grid, dv, rng=None):
    """
    Generate a randomized root-power noise spectrum due to the quantum
    fluctuations of a coherent state.

    This noise is independent of the field amplitude and is equivalent to
    the noise due to vacuum fluctuations.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    dv : float
        The frequency grid spacing.
    rng : np.random.Generator, optional
        A NumPy random number generator. Set `rng` to pass in an already
        initialized `Generator`, the default initializes a new `Generator`
        at each function call.

    Notes
    -----
    This **is not** shot noise (which is related to the collapse of coherent
    states onto discrete number states during photon detection), but the noise
    due to the quantum uncertainty of a coherent state's amplitude and phase.
    A coherent state :math:`|\\alpha\\rangle` is defined by a displacement
    :math:`\\alpha` and the set of number states :math:`|n\\rangle`:

    ..  math::
        |\\alpha\\rangle = e^{-\\frac{|\\alpha|^2}{2}}
            \sum_{n=0}^\\infty \\frac{\\alpha^n}{\sqrt{n!}} |n\\rangle

    The probability distribution :math:`P[\\alpha]` of finding a coherent
    state with displacement :math:`\\alpha`, given an average displacement
    :math:`\\beta`, is as follows:

    ..  math::
        &\\text{with } \\alpha = x_1 + i \\, x_2

        P[\\alpha] &= \\frac{1}{\\pi} |\\langle \\alpha | \\beta\\rangle|^2
            = \\frac{1}{\\pi} e^{-|\\alpha - \\beta|^2}

    where :math:`x_1` and :math:`x_2` are the root-photon normalized
    "amplitude" and "phase" (real and imaginary) field quadratures.

    Since the probability distribution is gaussian the noise is completely
    described by the variance of the two quadratures, which are scaled to the
    number of photons (:math:`N=\\alpha^2`). The combined noise from both
    quadratures gives a total variance of one photon per measurement:

    ..  math:: \\sigma_{x_1}^2 = \\sigma_{x_2}^2 = \\frac{1}{2}

    ..  math:: \\sigma_\\alpha^2 = \\sigma_{x_1}^2 + \\sigma_{x_2}^2 = 1

    The width of the probability distribution is independent of the average
    displacement of the coherent state, so the root-photon noise may be
    generated independently by sampling a standard normal distribution
    centered about zero mean. Also, since the Fourier transform of gaussian
    noise is also gaussian noise, the root-photon noise can be equivalently
    generated in either the time or frequency domains. Normalizing to the
    number of photons per measurement interval, the root photon noise for both
    quadratures becomes ``1/(2 * dt)**0.5`` for the time domain and
    ``1/(2 * dv)**0.5`` for frequency domain. The final root-power noise is
    found by multiplying the frequency domain root-photon noise by the square
    root of the photon energy associated with each bin's frequency.

    Returns
    -------
    a_v : ndarray of complex
        The randomly generated coherent state root-power noise.

    """
    if rng is None:
        rng = np.random.default_rng()

    v_grid = np.asarray(v_grid, dtype=float)
    n = v_grid.size
    a_v = ((h*v_grid)/(2*dv))**0.5 * (rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return a_v

def shift_v(v_grid, f_v, dv):
    pass #TODO: Fouier shift in frequency

def shift_t(t_grid, f_t, dt):
    pass #TODO: Fourier shift in time

def derivative_v(v_grid, f_v, n, t_ref=0):
    """
    Calculate the derivative of a frequency domain function using the Fourier
    method. This method is only strictly valid for input functions that have
    zero mean.

    The complementary time data is assumed to be of finite support,
    discontinuities in the frequency domain amplitude will manifest as ringing
    in the derivatives.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    f_v : array_like of complex
        The frequency domain function.
    n : float
        The order of the derivative. Positive orders correspond to derivatives
        and negative orders correspond to antiderivatives (integrals).
    t_ref : float, optional
        The grid reference time in the complementary time domain. The default
        is 0.

    Returns
    -------
    ndarray of complex

    """
    assert (len(v_grid) == len(f_v)), ("The frequency grid and frequency"
                                       " domain data must be the same length.")
    #---- Inverse Transform
    dv = np.diff(v_grid).mean()
    n_0 = len(v_grid)
    dt = 1/(n_0*dv)
    f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v), fsc=dt, overwrite_x=True))

    #---- Derivative
    t_grid = dt*(np.arange(n_0) - (n_0//2)) + t_ref

    dfdv_t = (-1j*2*pi*t_grid)**n * f_t
    dfdv_t[t_grid == 0] = 0

    #---- Transform
    dfdv_v = fft.fftshift(fft.fft(fft.ifftshift(dfdv_t), fsc=dt, overwrite_x=True))
    return dfdv_v

def derivative_t(t_grid, f_t, n, v_ref=0):
    """
    Calculate the derivative of a time domain function using the Fourier
    method. This method is only strictly valid for input functions that have
    zero mean.

    The complementary frequency data is assumed to be band-limited,
    discontinuities in the time domain amplitude will manifest as ringing in
    the derivatives.

    Parameters
    ----------
    t_grid : array_like of float
        The time grid.
    f_t : array_like of complex
        The time domain function.
    n : float
        The order of the derivative. Positive orders correspond to derivatives
        and negative orders correspond to antiderivatives (integrals).
    v_ref : float, optional
        The grid reference frequency in the complementary frequency domain.
        The default is 0.

    Returns
    -------
    ndarray of complex

    """
    assert (len(t_grid) == len(f_t)), ("The time grid and time domain data"
                                       " must be the same length.")
    #---- Transform
    n_0 = len(t_grid)
    dt = np.diff(t_grid).mean()
    dv = 1/(n_0*dt)
    if np.isrealobj(f_t) and v_ref==0:
        # Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t), fsc=dt)
        v_grid = dv*np.arange(len(f_v))
    else:
        # Complex Envelope Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt, overwrite_x=True))
        v_grid = dv*(np.arange(n_0) - (n_0//2)) + v_ref

    #---- Derivative
    dfdt_v = (+1j*2*pi*v_grid)**n * f_v
    dfdt_v[v_grid == 0] = 0

    #---- Inverse Transform
    if np.isrealobj(f_t) and v_ref==0:
        # Real-Valued Representation
        dfdt_t = fft.fftshift(fft.irfft(dfdt_v, fsc=dt, n=n_0))
    else:
        # Complex Envelope Representation
        dfdt_t = fft.fftshift(fft.ifft(fft.ifftshift(dfdt_v), fsc=dt, overwrite_x=True))
    return dfdt_t

def resample_v(v_grid, f_v, n):
    """
    Resample frequency domain data to the given number of points.

    The complementary time data is assumed to be of finite support, so the
    resampling is accomplished by adding or removing trailing and leading time
    bins. Discontinuities in the frequency domain amplitude will manifest as
    ringing when resampled.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid of the input data.
    f_v : array_like of complex
        The frequency domain data to be resampled.
    n : int
        The number of points at which to resample the input data. When the
        input corresponds to a real-valued time domain representation, this
        number is the number of points in the time domain.

    Returns
    -------
    v_grid : ndarray of float
        The resampled frequency grid.
    f_v : ndarray of real or complex
        The resampled frequency domain data.
    dv : float
        The spacing of the resampled frequency grid.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If the number of points is odd, there are an equal number of points on
    the positive and negative side of the time grid. If even, there is one
    extra point on the negative side.

    This method checks if the origin is contained in `v_grid` to determine
    whether real or complex transforms should be performed. In both cases the
    resampling is accomplished by removing trailing and leading time bins.

    For complex envelope representations, the returned frequency grid is
    defined symmetrically about its reference, as in the `TFGrid` class, and
    for real-valued representations the grid is defined starting at the origin.

    """
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    assert (len(v_grid) == len(f_v)), ("The frequency grid and frequency"
                                       " domain data must be the same length.")
    #---- Inverse Transform
    dv_0 = np.diff(v_grid).mean()
    if v_grid[0] == 0:
        assert np.isreal(f_v[0]), ("When the input is in the real-valued"
                                   " representation, the amplitude at the origin must be real.")

        # Real-Valued Representation
        if np.isreal(f_v[-1]):
            n_0 = 2*(len(v_grid)-1)
        else:
            n_0 = 2*(len(v_grid)-1) + 1
        dt_0 = 1/(n_0*dv_0)
        f_t = fft.fftshift(fft.irfft(f_v, fsc=dt_0, n=n_0))
    else:
        # Complex Envelope Representation
        n_0 = len(v_grid)
        dt_0 = 1/(n_0*dv_0)
        v_ref_0 = v_grid[n_0//2]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v), fsc=dt_0, overwrite_x=True))

    #---- Resample
    dn_n = n//2 - n_0//2 # leading time bins
    dn_p = (n-1)//2 - (n_0-1)//2 # trailing time bins
    if n > n_0:
        f_t = np.pad(f_t, (dn_n, dn_p), mode="constant", constant_values=0)
    elif n < n_0:
        f_t = f_t[-dn_n:n_0+dn_p]

    #---- Transform
    dt = 1/(n_0*dv_0)
    dv = 1/(n*dt)
    if v_grid[0] == 0:
        # Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t), fsc=dt)
        v_grid = dv*np.arange(len(f_v))
    else:
        # Complex Envelope Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt, overwrite_x=True))
        v_grid = dv*(np.arange(n) - (n//2))
        v_grid += v_ref_0

    #---- Construct ResampledV
    resampled = _ResampledV(v_grid=v_grid, f_v=f_v, dv=dv, dt=1/(n*dv))
    return resampled

def resample_t(t_grid, f_t, n):
    """
    Resample time domain data to the given number of points.

    The complementary frequency data is assumed to be band-limited, so the
    resampling is accomplished by adding or removing high frequency bins.
    Discontinuities in the time domain amplitude will manifest as ringing when
    resampled.

    Parameters
    ----------
    t_grid : array_like of float
        The time grid of the input data.
    f_t : array_like of real or complex
        The time domain data to be resampled.
    n : int
        The number of points at which to resample the input data.

    Returns
    -------
    t_grid : ndarray of float
        The resampled time grid.
    f_t : ndarray of real or complex
        The resampled time domain data.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If real, the resampling is accomplished by adding or removing the largest
    magnitude frequency components (both positive and negative). If complex,
    the input data is assumed to be analytic, so the resampling is accomplished
    by adding or removing the largest positive frequencies. This method checks
    the input data's type, not the magnitude of its imaginary component, to
    determine if it is real or complex.

    The returned time axis is defined symmetrically about the input's
    reference, such as in the `TFGrid` class.

    """
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    assert (len(t_grid) == len(f_t)), ("The time grid and time domain data"
                                       " must be the same length.")
    #---- Define Time Grid
    n_0 = len(t_grid)
    dt_0 = np.diff(t_grid).mean()
    t_ref_0 = t_grid[n_0//2]
    dv = 1/(n_0*dt_0)
    dt = 1/(n*dv)
    t_grid = dt*(np.arange(n) - (n//2))
    t_grid += t_ref_0

    #---- Resample
    if np.isrealobj(f_t):
        # Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t), fsc=dt_0)
        if (n > n_0) and not (n % 2):
            f_v[-1] /= 2 # renormalize aliased Nyquist component
        f_t = fft.fftshift(fft.irfft(f_v, fsc=dt, n=n))
    else:
        # Complex Envelope Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt_0, overwrite_x=True))
        if n > n_0:
            f_v = np.pad(f_v, (0, n-n_0), mode="constant", constant_values=0)
        elif n < n_0:
            f_v = f_v[:n]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v), fsc=dt, overwrite_x=True))

    #---- Construct ResampledT
    resampled = _ResampledT(t_grid=t_grid, f_t=f_t, dt=dt)
    return resampled


# %% Classes

class TFGrid():
    """
    Complementary grids defined over both time and frequency domains for the
    representation of analytic functions with complex-valued envelopes.

    The frequency grid is shifted and scaled such that the grid is aligned with
    the origin and contains only positive frequencies. The values given to the
    initializers are only targets and may be adjusted slightly. If necessary,
    the frequency step size will be decreased so that the grid is formed
    without any negative frequencies, fitting the desired number of points
    between (0, `v_max`].

    Parameters
    ----------
    n_points : int
        The number of grid points.
    v_max : float
        The target maximum frequency.
    dv : float
        The target frequency grid step size.
    v0 : float, optional
        The comoving frame reference frequency. The default selects the
        central frequency of the resulting grid.

    Notes
    -----
    For discrete Fourier transforms (DFT), the frequency step multiplied by
    the time step is always equal to the reciprocal of the total number of
    points::

        dt*dv == 1/n

    Each grid point represents the midpoint of a bin that extends 0.5 grid
    spacings in both directions.

    Aligning the frequency grid to the origin facilitates calculations using
    real Fourier transforms, which have grids that start at zero frequency. The
    `rtf_grids` method and the `rn_range` and `rn_slice` attributes are useful
    when transitioning between the analytic, complex envelope representation
    of this class to the real-valued representation.

    The comoving frame reference frequency `v0` does not affect the definition
    of the grids but defines the frequency at which the comoving frame is
    referenced.

    By definition of the DFT, the time and frequency grids must range
    symmetrically about the origin, with the time grid incrementing in unit
    steps and the frequency grid in steps of ``1/n``. The grids of the `TFGrid`
    class are scaled and shifted such that they represent absolute time or
    frequency values. The scaling is accomplished by setting the forward scale
    parameter of the Fourier transforms to ``dt``. The `v_ref` and `t_ref`
    variables describe the amount that the `TFGrid` grids need to be shifted
    to come into alignment with the origins of the grids implicitly defined by
    the DFT.

    """

    def __init__(self, n_points, v_max, dv, v0=None):
        """
        Initialize a time and frequency grid given a target maximum frequency,
        a target frequency step size, and the total number of grid points.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_max : float
            The target maximum frequency.
        dv : float
            The target frequency step size.
        v0 : float, optional
            The comoving frame reference frequency. The default selects the
            central frequency of the resulting grid.

        """
        assert isinstance(n_points, (int, np.integer)), "The number of points must be an integer."
        assert (n_points > 1),  "The number of points must be greater than 1."
        assert (dv > 0), "The frequency grid step size must be greater than 0."
        assert (v_max > 0), "The target maximum frequency must be greater than 0."

        #---- Align Frequency Grid
        v_max_index = int(round(np.modf(v_max / dv)[1]))
        v_min_index = v_max_index - (n_points-1)
        if v_min_index < 1:
            v_max_index = n_points
            v_min_index = 1
            dv = v_max/v_max_index
        self._rn_range = np.array([v_min_index, v_max_index])
        self._rn_slice = slice(self.rn_range.min(), self.rn_range.max()+1)
        self._n = n_points

        #---- Define Frequency Grid
        self._dv = dv
        self._v_grid = fft.ifftshift(self.dv*(np.arange(self.n) + v_min_index))
        self._v_ref = self._v_grid[0]
        self._v_window = self.n*self.dv
        if v0 is None:
            self.v0 = self.v_grid[self.n//2] # same as v_ref
        else:
            self.v0 = v0

        #---- Define Complex Time Grid
        self._dt = 1/(self.n*self.dv)
        self._t_grid = fft.ifftshift(self.dt*(np.arange(n_points) - (self.n//2)))
        self._t_ref = self._t_grid[0]
        self._t_window = self.n*self.dt

        #---- Define Real Time and Frequency Grids
        self.rtf_grids(n_harmonic=1, update=True)

    #---- Class Methods
    @classmethod
    def FromFreqRange(cls, n_points, v_min, v_max, v0=None):
        """
        Initialize a time and frequency grid given a target minimum and maximum
        frequency and the total number of grid points.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float, optional
            The comoving frame reference frequency. The default selects the
            central frequency of the resulting grid.

        """
        assert (v_max > v_min), ("The target maximum frequency must be greater"
                                 " than the target minimum frequency.")
        dv = (v_max - v_min)/(n_points-1)
        return cls(n_points, v_max, dv, v0=v0)

    @classmethod
    def FromTimeWindowAndFreq(cls, n_points, t_window, v0):
        """
        Initialize a time and frequency grid given a target time window, a
        target center frequency, and the total number of grid points.

        If there are too many points to place `v0` at the center of the grid
        the excess number is added at higher frequencies.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        t_window : float
            The target time window.
        v0 : float
            The target center frequency, which is also taken as the comoving
            frame reference frequency.

        """
        assert (t_window > 0), "The target time window must be greater than 0."
        assert (v0 > 0), "The target center frequency must be greater than 0."

        dt = t_window/n_points
        dv = 1/(n_points*dt)

        v_min = v0 - (n_points//2) * dv
        v_min_index = int(round(np.modf(v_min / dv)[1]))
        if v_min_index < 1:
            v_max_index = n_points
        else:
            v_max_index = v_min_index + (n_points-1)
        v_max = dv * v_max_index
        return cls(n_points, v_max, dv, v0=v0)

    #---- General Properties
    @property
    def n(self):
        """
        The number of grid points in the complex envelope representation.

        This value is the same for both the time and frequency grids.

        Returns
        -------
        int

        """
        return self._n

    @property
    def rn(self):
        """
        The number of grid points in the real-valued time domain
        representation.

        Returns
        -------
        int

        """
        return self._rn

    @property
    def rn_range(self):
        """
        The minimum and maximum indices of the origin contiguous frequency
        grid associated with the real-valued time domain representation that
        correspond to the first and last points of the analytic frequency grid
        associated with the complex envelope time domain representation.

        These values are useful for indexing and constructing frequency grids
        for applications with real DFTs.

        Returns
        -------
        ndarray of float

        """
        return self._rn_range

    @property
    def rn_slice(self):
        """
        A slice object that indexes the origin contiguous frequency grid
        associated with the real-valued time domain representation onto the
        analytic frequency grid associated with the complex envelope time
        domain representation.

        This is useful for indexing and constructing frequency gridded arrays
        for applications with real DFTs. It is assumed that the arrays are
        arranged such that the frequency coordinates are monotonically ordered.

        Returns
        -------
        slice

        """
        return self._rn_slice

    @property
    def v0(self):
        """
        The comoving frame reference frequency.

        Returns
        -------
        float

        """
        return self._v0
    @v0.setter
    def v0(self, v0):
        assert (v0 > 0), "The comoving frame reference frequency must be greater than 0."
        self._v0_idx = np.argmin(np.abs(self.v_grid - v0))
        self._v0 = self.v_grid[self.v0_idx]

    @property
    def v0_idx(self):
        """
        The array index of the comoving frame’s reference frequency.

        Returns
        -------
        int

        """
        return self._v0_idx

    #---- Frequency Grid Properties
    @property
    def v_grid(self):
        """
        The frequency grid in the complex envelope representation, with units
        of ``Hz``.

        The frequency grid is aligned to the origin and contains only positive
        frequencies.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._v_grid)

    @property
    def v_ref(self):
        """
        The grid reference frequency in the complex envelope representation.

        This is the frequency offset between `v_grid` and the origin of the
        frequency grid implicitly defined by a DFT with `n` points.

        Returns
        -------
        float

        """
        return self._v_ref

    @property
    def dv(self):
        """
        The frequency grid step size in the complex envelope representation.

        Returns
        -------
        float

        """
        return self._dv

    @property
    def v_window(self):
        """
        The span of the frequency grid in the complex envelope representation.

        This is equal to the number of grid points times the frequency grid
        step size.

        Returns
        -------
        float

        """
        return self._v_window

    #---- Time Grid Properties
    @property
    def t_grid(self):
        """
        The time grid in the complex envelope representation, with units of
        ``s``.

        The time grid is aligned symmetrically about the origin.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._t_grid)

    @property
    def t_ref(self):
        """
        The grid reference time in the complex envelope representation.

        This is the time offset between `t_grid` and the origin of the time
        grid implicitly defined by a DFT with `n` points.

        Returns
        -------
        float

        """
        return self._t_ref

    @property
    def dt(self):
        """
        The time grid step size in the complex envelope representation.

        This is the differential for Fourier transforms. Multiplying the
        integrand of the transform by this factor will preserve the integrated
        absolute squared magnitude::

            a_v = fft.fft(a_t, fsc=dt)
            np.sum(np.abs(a_t)**2 * dt) == np.sum(np.abs(a_v)**2 *dv)

        Returns
        -------
        float

        """
        return self._dt

    @property
    def t_window(self):
        """
        The span of the time grid in the complex envelope representation.

        This is equal to the number of grid points time the time grid step
        size.

        Returns
        -------
        float

        """
        return self._t_window

    #---- Real Time/Frequency Grid Properties
    @property
    def rv_grid(self):
        """
        The origin contiguous frequency grid for the real-valued time domain
        representation, with units of ``Hz``.

        Returns
        -------
        ndarray of float

        """
        return self._rv_grid

    @property
    def rv_ref(self):
        """
        The grid reference frequency in the real-valued time domain
        representation.

        Returns
        -------
        float

        """
        return self._rv_ref

    @property
    def rdv(self):
        """
        The frequency grid step size in the real-valued time domain
        representation.

        This is equal to the frequency grid step size in the complex envelope
        representation.

        Returns
        -------
        float

        """
        return self._dv

    @property
    def rv_window(self):
        """
        The span of the frequency grid in the real-valued time domain
        representation.

        Returns
        -------
        float

        """
        return self._rv_window

    @property
    def rt_grid(self):
        """
        The time grid in the real-valued time domain representation, with
        units of ``s``.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._rt_grid)

    @property
    def rt_ref(self):
        """
        The grid reference time in the real-valued time domain representation.

        Returns
        -------
        float

        """
        return self._rt_ref

    @property
    def rdt(self):
        """
        The time grid step size in the real-valued time domain representation.

        Returns
        -------
        float

        """
        return self._rdt

    @property
    def rt_window(self):
        """
        The span of the time grid in the real-valued time domain
        representation.

        Returns
        -------
        float

        """
        return self._rt_window

    def rtf_grids(self, n_harmonic=1, fast_n=True, update=True):
        """
        Complementary grids defined over both time and frequency domains for
        the representation of analytic functions with real-valued amplitudes.

        The frequency grid contains the origin and positive frequencies. The
        `n_harmonic` parameter determines the number of harmonics the
        resulting time grid supports without aliasing. In order to maintain
        efficient DFT behavior, the number of points can be extended further based
        on the output of `scipy.fft.next_fast_len`. These grids are suitable
        for use with real DFTs, see `fft.rfft` and `fft.irfft`.

        Parameters
        ----------
        n_harmonic : int, optional
            The harmonic support of the generated grids. The default is 1, the
            fundamental harmonic.
        fast_n : bool, optional
            A parameter that determines whether the length of the
            array is extended up to the next fast fft length. The default is
            to extend.
        update : bool, optional
            A parameter that determines whether to update the real time
            and frequency grids of the parent object with the results of this method.
            The default is to update.

        Returns
        -------
        n : int
            The number of grid points.
        v0 : float
            The comoving frame reference frequency.
        v_grid : array of float
            The origin contiguous frequency grid.
        v_ref : float
            The grid reference frequency.
        dv : float
            The frequency grid step size.
        v_window : float
            The span of the frequency grid.
        t_grid : array of float
            The time grid.
        t_ref : float
            The grid reference time.
        dt : float
            The time grid step size.
        t_window : float
            The span of the time grid.

        Notes
        -----
        Multiplication of functions in the time domain (an operation intrinsic
        to nonlinear optics) is equivalent to convolution in the frequency
        domain and vice versa. The support of a convolution is the sum of the
        support of its parts. Thus, in order to avoid aliasing, 2nd and 3rd
        order processes in the time domain need support up to the 2nd and 3rd
        harmonics in the frequency domain.

        To avoid dealing with case-specific amplitude scale factors when
        transforming between complex envelope and real-valued representations
        the frequency grid for complex-valued function must not contain the
        origin and there must be enough points in the real-valued
        representation to avoid aliasing the Nyquist frequency of the complex
        envelope representation. The initializer of this class enforces the
        first condition, the frequency grid starts at minimum one step size
        away from the origin, and this method enforces the second by making
        the minimum number of points odd if at the first harmonic.

        The transformation between representations is performed as in the
        following example, with `tf` an instance of the `TFGrid` class, `rtf`
        the output of this method, `a_v` the spectrum of a complex-valued
        envelope defined over `v_grid`, `ra_v` the spectrum of the real-valued
        function defined over `rtf.v_grid`, and `ra_t` the real-valued
        function defined over `rtf.t_grid`. The ``1/2**0.5`` scale factor
        between `a_v` and `ra_v` preserves the integrated squared magnitude in
        the time domain::

            rtf = tf.rtf_grids()
            ra_v = np.zeros_like(rtf.v_grid, dtype=complex)
            ra_v[tf.rn_slice] = 2**-0.5 * a_v
            ra_t = fft.irfft(ra_v, fsc=rtf.dt, n=rtf.n)
            np.sum(ra_t**2 * rtf.dt) == np.sum(np.abs(a_v)**2 * tf.dv)

        """
        assert (n_harmonic >= 1), "The harmonic support must be atleast 1."
        #---- Number of Points
        target_n_v = self.rn_range.max()*n_harmonic
        if n_harmonic == 1:
            target_n_t = 2*target_n_v - 1 # odd
        else:
            target_n_t = 2*(target_n_v - 1) # even
        if fast_n:
            n = fft.next_fast_len(round(target_n_t))
        else:
            n = target_n_t
        n_v = n//2 + 1

        #---- Define Frequency Grid
        v_grid = self.dv*np.arange(n_v)
        v_ref = v_grid[0]

        #---- Define Time Grid
        dt = 1/(n*self.dv)
        t_grid = dt*(np.arange(n) - (n//2))
        t_ref = t_grid[n//2]

        #---- Construct RTFGrid
        rtf_grids = _RTFGrid(
            n=n, v0=self.v0,
            v_grid=v_grid, v_ref=v_ref, dv=self.dv, v_window=n_v*self.dv,
            t_grid=t_grid, t_ref=t_ref, dt=dt, t_window=n*dt)

        if update:
            self._rn = rtf_grids.n

            # Frequency Grid
            self._rv_grid = rtf_grids.v_grid
            self._rv_ref = rtf_grids.v_ref
            self._rv_window = rtf_grids.v_window

            # Time Grid
            self._rt_grid = fft.ifftshift(rtf_grids.t_grid)
            self._rt_ref = rtf_grids.t_ref
            self._rdt = rtf_grids.dt
            self._rt_window = rtf_grids.t_window
        return rtf_grids
