# -*- coding: utf-8 -*-
"""
Routines and classes for representing optical pulses in the time and frequency
domains.

Notes
-----
The public facing routines and properties of the defined classes have inputs
and outputs that are aranged such that the coordinate arrays are monotonicallyl
ordered. Many of the associated private methods and properties are arranged in
the standard fft order.
"""

__all__ = ["resample_v", "resample_t", "TFGrid", "Pulse"]


# %% Imports

import collections
import copy

import numpy as np
from scipy import constants, fft


# %% Constants

pi = constants.pi


# %% Routines

def derivative_v(v_grid, f_v, n, t_ref=0):
    """
    Calculate the derivative of a frequency domain function using the Fourier
    method. This method is only strictly valid for input functions that have
    zero mean.

    The complementary time data is assumed to be of finite support,
    discontinuities in the frequency domain amplitude will manifest as ringing
    in the derivatives.  Because taking the derivative involes multiplying the
    time domain amplitudes by a complex number, this method does not support
    spectra input in the real-valued representation.

    Parameters
    ----------
    v_grid : array of float
        The frequency grid.
    f_v : array of complex
        The frequency domain function.
    n : float
        The order of the derivative. Positive orders correspond to derivatives
        and negative orders correspond to antiderivatives (integrals).
    t_ref : float, optional
        The grid reference time in the complementary time domain. The default
        is 0.

    Returns
    -------
    array of complex
    """
    assert (len(v_grid) == len(f_v)), "The frequency grid and frequency domain data must be the same length."

    #--- Inverse Transform
    dv = np.diff(v_grid).mean()
    n_0 = len(v_grid)
    dt = 1/(n_0*dv)
    f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v)*n_0*dv))

    #--- Derivative
    n_2 = n_0//2
    t_grid = dt*(np.arange(n_0) - n_2) + t_ref

    dfdv_t = (-1j*2*pi*t_grid)**n * f_t
    dfdv_t[t_grid == 0] = 0

    #--- Transform
    dfdv_v = fft.fftshift(fft.fft(fft.ifftshift(dfdv_t)*dt))
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
    t_grid : array of float
        The time grid.
    f_t : array of complex
        The time domain function.
    n : float
        The order of the derivative. Positive orders correspond to derivatives
        and negative orders correspond to antiderivatives (integrals).
    v_ref : float, optional
        The grid reference frequency in the complementary frequency domain. The
        default is 0.

    Returns
    -------
    array of complex
    """
    assert (len(t_grid) == len(f_t)), "The time grid and time domain data must be the same length."

    #--- Transform
    n_0 = len(t_grid)
    dt = np.diff(t_grid).mean()
    dv = 1/(n_0*dt)
    if np.isrealobj(f_t) and v_ref==0:
        #--- Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t)*dt)
        v_grid = dv*np.arange(len(f_v))
    else:
        #--- Complex Envelope Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t)*dt))
        n_2 = n_0//2
        v_grid = dv*(np.arange(n_0) - n_2) + v_ref

    #--- Derivative
    dfdt_v = (+1j*2*pi*v_grid)**n * f_v
    dfdt_v[v_grid == 0] = 0

    #--- Inverse Transform
    if np.isrealobj(f_t) and v_ref==0:
        #--- Real-Valued Representation
        dfdt_t = fft.fftshift(fft.irfft(dfdt_v*n_0*dv, n=n_0))
    else:
        #--- Complex Envelope Representation
        dfdt_t = fft.fftshift(fft.ifft(fft.ifftshift(dfdt_v)*n_0*dv))
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
    v_grid : array of float
        The frequency grid of the input data.
    f_v : array of complex
        The frequency domain data to be resampled.
    n : int
        The number of points at which to resample the input data. When the
        input corresponds to a real-valued time domain representation, this
        number is the number of points in the time domain.

    Returns
    -------
    v_grid : array of float
        The resampled frequency grid.
    f_v : array of real or complex
        The resampled frequency domain data.
    dv : float
        The spacing of the resampled frequency grid.
    ndv : float
        The spacing of the resampled frequency grid multiplied by the
        number of points.

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
    for real-valued the grid is defined starting at the origin.
    """
    assert (len(v_grid) == len(f_v)), "The frequency grid and frequency domain data must be the same length."
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    results = {}

    #--- Inverse Transform
    dv_0 = np.diff(v_grid).mean()
    if v_grid[0] == 0:
        assert np.isreal(f_v[0]), "When the input is in the real-valued representation, the amplitude at the origin must be real."

        #--- Real-Valued Representation
        if np.isreal(f_v[-1]):
            n_0 = 2*(len(v_grid)-1)
        else:
            n_0 = 2*(len(v_grid)-1) + 1
        f_t = fft.fftshift(fft.irfft(f_v*n_0*dv_0, n=n_0))
    else:
        #--- Complex Envelope Representation
        n_0 = len(v_grid)
        v_ref_0 = fft.ifftshift(v_grid)[0]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v)*n_0*dv_0))

    #--- Resample
    dn_n = n//2 - n_0//2 # leading time bins
    dn_p = (n-1)//2 - (n_0-1)//2 # trailing time bins
    if n > n_0:
        f_t = np.pad(f_t, (dn_n, dn_p), mode="constant", constant_values=0)
    elif n < n_0:
        start_idx = -dn_n
        stop_idx = n_0+dn_p
        f_t = f_t[start_idx:stop_idx]

    #--- Transform
    dt = 1/(n_0*dv_0)
    dv = 1/(n*dt)
    if v_grid[0] == 0:
        #--- Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t)*dt)
        v_grid = dv*np.arange(len(f_v))
    else:
        #--- Complex Envelope Representation
        n_2 = n//2
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t)*dt))
        v_grid = dv*(np.arange(n) - n_2)
        v_grid += v_ref_0
    results["v_grid"] = v_grid
    results["f_v"] = f_v

    #--- Construct ResampledV
    results["dv"] = dv
    results["ndv"] = n*dv

    resampled = collections.namedtuple("ResampledV", results.keys())(**results)
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
    t_grid : array of float
        The time grid of the input data.
    f_t : array of real or complex
        The time domain data to be resampled.
    n : int
        The number of points at which to resample the input data.

    Returns
    -------
    t_grid : array of float
        The resampled time grid.
    f_t : array of real or complex
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
    assert (len(t_grid) == len(f_t)), "The time grid and time domain data must be the same length."
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    results = {}

    #--- Define Time Grid
    n_0 = len(t_grid)
    dt_0 = np.diff(t_grid).mean()
    t_ref_0 = fft.ifftshift(t_grid)[0]
    dv = 1/(n_0*dt_0)
    dt = 1/(n*dv)
    n_2 = n//2
    t_grid = dt*(np.arange(n) - n_2)
    t_grid += t_ref_0
    results["t_grid"] = t_grid

    #--- Resample
    if np.isrealobj(f_t):
        #--- Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t)*dt_0)
        if (n > n_0) and not (n % 2):
            f_v[-1] /= 2 # renormalize aliased Nyquist component
        f_t = fft.fftshift(fft.irfft(f_v*n*dv, n=n))
    else:
        #--- Complex Envelope Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t)*dt_0))
        if n > n_0:
            f_v = np.pad(f_v, (0, n-n_0), mode="constant", constant_values=0)
        elif n < n_0:
            f_v = f_v[:n]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v)*n*dv))
    results["f_t"] = f_t

    #--- Construct ResampledT
    results["dt"] = dt

    resampled = collections.namedtuple("ResampledT", results.keys())(**results)
    return resampled


# %% Classes

class TFGrid():
    """
    Complementary grids, defined over the time and frequency domains, for
    representing analytic functions with complex-valued envelopes.

    The frequency grid is shifted and scaled such that the grid is aligned with
    the origin and contains only positive frequencies. The values given to the
    initializers are only targets and may be slightly adjusted. If necessary,
    the frequency step size will be decreased so that the grid is formed
    without any negative frequencies, fitting the desired number of points
    between (0, `v_max`].

    Parameters
    ----------
    v_max : float
        The target maximum frequency.
    dv : float
        The target frequency grid step size.
    n_points : int
        The number of grid points.
    v0 : float, optional
        The retarded frame reference frequency. The default (``None``) is the
        average of the frequency grid.

    Attributes
    ----------
    n
    rn
    rn_range
    rn_slice
    v0
    v0_idx
    v_grid
    v_ref
    dv
    ndv
    v_window
    t_grid
    t_ref
    dt
    t_window
    rv_grid
    rv_ref
    rdv
    rndv
    rv_window
    rt_grid
    rt_ref
    rdt
    rt_window

    Methods
    -------
    from_freq_range
    from_time_and_freq

    Notes
    -----
    For discrete Fourier transforms, the frequency step multiplied with the
    time step is always equal to the reciprocal of the total number of points::

        dt*dv == 1/n

    The grid points represent the midpoint of a bin that extends 0.5 grid
    spacings in both directions.

    Aligning the frequency grid to the origin facilitates calculations using
    real fourier transforms, which have grids that start at zero frequency. The
    `rtf_grids` method and the `rn_range` and `rn_slice` attributes are useful
    when transitioning between the analyitic, complex envelope representation
    of this class to the real-valued representation.

    The retarded frame reference frequency `v0` does not affect the definition
    of the grids but defines the frequency at which the comoving, retarded
    frame is referenced.

    By definition of the DFT, the time and frequency grids must range
    symmetrically about the origin, with the time grid incrementing in unit
    steps and the frequency grid in steps of 1/`n`. The grids of the `TFGrid`
    class are scaled and shifted such that they are representative of absolute
    time or frequency values. The scaling is accomplished by multiplying the
    arguments of the transform and inverse (the time/spectral densities) by
    `dt` and `ndt`. The `v_ref` and `t_ref` variables describe the amount that
    the `TFGrid` grids need to be shifted to come into alignment with the
    origins of the grids implicitly defined by the DFT.
    """

    def __init__(self, v_max, dv, n_points, v0=None):
        """
        Initialize a time and frequency grid given a target maximum frequency,
        a target frequency step size, and the total number of grid points.

        Parameters
        ----------
        v_max : float
            The target maximum frequency.
        dv : float
            The target frequency step size.
        n_points : int
            The number of grid points.
        v0 : float, optional
            The retarded frame reference frequency. The default (``None``) is
            the average of the resulting frequency grid.
        """
        assert (dv > 0), "The frequency grid step size must be greater than 0."
        assert (v_max > 0), "The target maximum frequency must be greater than 0."
        assert isinstance(n_points, (int, np.integer)), "The number of points must be an integer."
        assert (n_points > 1),  "The number of points must be greater than 1."


        #--- Align Frequency Grid
        v_max_index = int(round(np.modf(v_max / dv)[1]))
        v_min_index = v_max_index - (n_points-1)
        if (v_min_index < 1):
            v_max_index = n_points
            v_min_index = 1
            dv = v_max/v_max_index
        self._rn_range = np.array([v_min_index, v_max_index])
        self._rn_slice = slice(self.rn_range.min(), self.rn_range.max()+1)
        self._n = n_points

        #--- Define Frequency Grid
        self._dv = dv
        self._ndv = self.n*self.dv
        self._v_grid = fft.ifftshift(self.dv*(np.arange(self.n) + v_min_index))
        self._v_ref = self._v_grid[0]
        self._v_window = self.n*self.dv
        if v0 is None:
            self.v0 = self.v_grid.mean()
        else:
            self.v0 = v0

        #--- Define Complex Time Grid
        n_2 = self.n // 2
        self._dt = 1/(self.n*self.dv)
        self._t_grid = fft.ifftshift(self.dt*(np.arange(n_points) - n_2))
        self._t_ref = self._t_grid[0]
        self._t_window = self.n*self.dt

        #--- Define Real Time and Frequency Grids
        rtf = self.rtf_grids(n_harmonic=1)
        self._rn = rtf.n

        # Frequency Grid
        self._rv_grid = rtf.v_grid
        self._rv_ref = rtf.v_ref
        self._rndv = rtf.ndv
        self._rv_window = rtf.v_window

        # Time Grid
        self._rt_grid = fft.ifftshift(rtf.t_grid)
        self._rt_ref = rtf.t_ref
        self._rdt = rtf.dt
        self._rt_window = rtf.t_window

    #--- Class Methods
    @classmethod
    def from_freq_range(cls, v_min, v_max, n_points, v0=None):
        """
        Initialize a time and frequency grid given a target minimum and maximum
        frequency and the total number of grid points.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float, optional
            The retarded frame reference frequency. The default (``None``) is
            the average of the resulting frequency grid.
        """
        assert (v_max > v_min), "The target maximum frequency must be greater than the target minimum frequency."
        assert (n_points > 1), "The number of points must be greater than 1."

        dv = (v_max - v_min)/(n_points-1)

        return cls(v_max, dv, n_points, v0=v0)

    @classmethod
    def from_time_and_freq(cls, t_window, v0, n_points):
        """
        Initialize a time and frequency grid given a target time window, a
        target center frequency, and the total number of grid points.

        Parameters
        ----------
        t_window : float
            The target time window.
        v0 : float
            The target center frequency, which is also taken as the retarded
            frame reference frequency.
        n_points : int
            The number of grid points.
        """
        assert (t_window > 0), "The target time window must be greater than 0."
        assert (v0 > 0), "The target center frequency must be greater than 0."
        assert (n_points > 1), "The number of points must be greater than 1."

        dt = t_window/n_points
        dv = 1/(n_points*dt)
        v_max = v0 + 0.5(n_points-1)*dv
        return cls(v_max, dv, n_points, v0=v0)

    #--- General Properties
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
        An array containing the indices of an origin contiguous frequency grid
        that correspond to the minimum and maximum of the frequency grid
        defined in this class.

        These values are useful for indexing and constructing frequency grids
        for applications with real DFTs.

        Returns
        -------
        array of float
        """
        return self._rn_range

    @property
    def rn_slice(self):
        """
        A slice object that indexes the frequency grid defined in this class
        onto an origin contiguous frequency grid.

        This is useful for indexing and constructing frequency gridded arrays
        for applications with real DFTs. It is assumed that the arrays are
        aranged such that the frequency coordinates are monotonically ordered.

        Returns
        -------
        slice object
        """
        return self._rn_slice

    @property
    def v0(self):
        """
        The retarded frame reference frequency.

        Set this property to update this reference frequency for future
        calculations.

        Returns
        -------
        float
        """
        return self._v0
    @v0.setter
    def v0(self, v0):
        assert (v0 > 0), "The retarded frame reference frequency must be greater than 0."
        self._v0_idx = np.argmin(np.abs(self.v_grid - v0))
        self._v0 = self.v_grid[self.v0_idx]

    @property
    def v0_idx(self):
        """
        The index location of the retarded fram reference frequency.
        """
        return self._v0_idx

    #--- Frequency Grid Properties
    @property
    def v_grid(self):
        """
        The frequency grid in the complex envelope representation.

        The frequency grid is aligned to the origin and contains only positive
        frequencies.

        Returns
        -------
        array of float
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
    def ndv(self):
        """
        The number grid points times the frequency grid step size in the
        complex envelope representation.

        This is the differential for inverse fourier transforms with
        normalization factor of 1/`n`. Multiplying the integrand of the inverse
        transform by this factor will preserve the integrated absolute squared
        magnitude::

            a_t = fft.ifft(fft.ifftshift(a_v)*ndv)
            np.sum(np.abs(a_v)**2 * dv) == np.sum(np.abs(a_t)**2 *dt)

        Returns
        -------
        float
        """
        return self._ndv

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

    #--- Time Grid Properties
    @property
    def t_grid(self):
        """
        The time grid in the complex envelope representation.

        The time grid is aligned symmetrically about the origin.

        Returns
        -------
        array of float
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

        This is the differential for fourier transforms with normalization
        factor of 1. Multiplying the integrand of the transform by this factor
        will preserve the integrated absolute squared magnitude::

            a_v = fft.fft(a_t*dt)
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

    #--- Real Time/Frequency Grid Properties
    @property
    def rv_grid(self):
        """
        The origin contiguous frequency grid for the real-valued time domain
        representation.

        Returns
        -------
        array of float
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
    def rndv(self):
        """
        The number grid points times the frequency grid step size in the
        real-valued time domain representation.

        Returns
        -------
        float
        """
        return self._rndv

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
        The time grid in the real-valued time domain representation.

        Returns
        -------
        array of float
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

    #--- Methods
    def rtf_grids(self, n_harmonic=1, fast_n=True):
        """
        Complementary grids defined over the time and frequency domains for
        representing analytic functions with real-valued amplitudes.

        The frequency grid contains the origin and positive frequencies. The
        `n_harmonic` parameter determines the number of harmonics the resulting
        time grid supports without aliasing. In order to maintain efficient
        DFT behavior, the number of points is further extended based on the
        output of `scipy.fft.next_fast_len`. These grids are suitable for use
        with real DFTs, see `numpy.fft.rfft` or `scipy.fft.rfft`.

        Parameters
        ----------
        n_harmonic : int, optional
            The harmonic support of the generated grids. The default is 1, the
            fundamental harmonic.
        fast_n : bool, optional
            A parameter that determines whether or not to the length of the
            array is extended up to the next fast fft length. The defualt is to
            extend.

        Returns
        -------
        n : int
            The number of grid points.
        v0 : float
            The retarded frame reference frequency.
        v_grid : array of float
            The origin contiguous frequency grid.
        v_ref : float
            The grid reference frequency.
        dv : float
            The frequency grid step size.
        ndv : float
            The number grid points times the frequency grid step size.
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
        support of its parts. Thus, 2nd and 3rd order processes in the time
        domain need support up to the 2nd and 3rd harmonics in the frequency
        domain in order to not alias.

        To avoid dealing with amplitude scale factors when changing between
        complex envelope and real-valued representations the frequency grid for
        the complex-valued function must not contain the origin and their must
        be enough points in the real-valued representation to place the Nyquist
        frequency above the maximum frequency of the complex envelope
        representation. The initializer of this class enforces the first
        condition, the frequency grid starts at minimum one step size away from
        the origin, and this method enforces the second by making the minimum
        number of points odd if at the first harmonic.

        The transformation between representations is performed as in the
        following example, with `tf` an instance of the `TFGrid` class, `rtf`
        the output of this method, `a_v` the spectrum of a complex-valued
        envelope defined over `v_grid`, `ra_v` the spectrum of the real-valued
        function defined over `rtf.v_grid`, and `ra_t` the real-valued function
        defined over `rtf.t_grid`. The ``1/2**0.5`` scale factor between `a_v`
        and `ra_v` preserves the integrated absolute squared magnitude::

            rtf = tf.rtf_grids()
            ra_v = np.zeros_like(rtf.v_grid, dtype=complex)
            ra_v[tf.rn_slice] = 2**-0.5 * a_v
            ra_t = fft.irfft(ra_v*rtf.ndv, n=rtf.n)
            np.sum(ra_t**2 * rtf.dt) == np.sum(np.abs(a_v)**2 * tf.dv)
        """
        assert isinstance(n_harmonic, (int, np.integer)), "The harmonic support must be an integer."
        assert (n_harmonic > 0), "The harmonic support must be greater than 0."
        results = {}

        #--- Number of Points
        target_n_v = self.rn_range.max()*n_harmonic
        if n_harmonic == 1:
            target_n_t = 2*target_n_v - 1 # odd
        else:
            target_n_t = 2*(target_n_v - 1) # even

        if fast_n:
            n = fft.next_fast_len(target_n_t)
        else:
            n = target_n_t

        n_v = n//2 + 1
        results["n"] = n

        #--- Define Frequency Grid
        v_grid = self.dv*np.arange(n_v)
        v_ref = v_grid[0]
        results["v0"] = self.v0
        results["v_grid"] = v_grid
        results["v_ref"] = v_ref
        results["dv"] = self.dv
        results["ndv"] = n*self.dv
        results["v_window"] = n_v*self.dv

        #--- Define Time Grid
        n_2 = n//2
        dt = 1/(n*self.dv)
        t_grid = dt*(np.arange(n) - n_2)
        t_ref = fft.ifftshift(t_grid)[0]
        results["t_grid"] = t_grid
        results["t_ref"] = t_ref
        results["dt"] = dt
        results["t_window"] = n*dt

        #--- Construct RTFGrid
        rtf_grids = collections.namedtuple("RTFGrid", results.keys())(**results)

        return rtf_grids


class Pulse(TFGrid):
    """
    Optical power and complex root-power envelopes defined over complementary
    time and frequency grids.

    The default initializer is the same as `TFGrid` and creates a `Pulse`
    object with no optical power. Set one of four attributes `a_v`, `p_v`,
    `a_t`, or `p_t` to populate the optical spectrum and
    envelope.

    Parameters
    ----------
    v_max : float
        The target maximum frequency.
    dv : float
        The target frequency grid step size.
    n_points : int
        The number of grid points.
    v0 : float, optional
        The retarded frame reference frequency. The default (``None``) is the
        average of the resulting frequency grid.

    Attributes
    ----------
    a_v
    p_v
    phi_v
    tg_v
    a_t
    p_t
    phi_t
    vg_t
    ra_t
    rp_t
    e_p

    Methods
    -------
    from_TFGrid
    from_spectrum
    gaussian_pulse
    sech2_pulse
    parabolic_pulse
    lorentzian2_pulse
    cw_light

    Notes
    -----
    See `TFGrid` for other attributes.

    The power spectrum and power envelope are normalized to the pulse energy
    `e_p`::

        e_p == np.sum(p_v*dv) == np.sum(p_t*dt) == np.sum(rp_t*rdt)

    The amplitude of the root-power spectrum of the complex envelope is a
    factor of ``2**0.5`` larger than the root-power spectrum of the real-valued
    representation. This difference is due to the real DFT being only the
    positive side of a double-sided spectrum. When transforming between the two
    representations use the following normalization::

        a_v = 2**0.5 * ra_v[rn_slice]
        ra_v[rn_slice] = 2**-0.5 * a_v

    Setting the `p_v`, `phi_v`, `p_t`, or `phi_t` attributes requires that the
    `a_v` or `a_t` attributes have already been initialized.
    """

    def __init__(self, v_max, dv, n_points, v0=None):
        """
        Initialize a time and frequency grid given a target maximum frequency,
        a target frequency step size, and the total number of grid points, and
        populate with a zero amplitude power spectrum.

        Set one of six attributes `a_v`, `p_v`, `phi_v`, `a_t`, `p_t`, or
        `phi_t` to populate the optical spectrum and envelope.

        Parameters
        ----------
        v_max : float
            The target maximum frequency.
        dv : float
            The target frequency grid step size.
        n_points : int
            The number of grid points.
        v0 : float, optional
            The retarded frame reference frequency. The default (``None``) is
            the average of the resulting frequency grid.
        """
        #--- Construct TF Grids
        super().__init__(v_max, dv, n_points, v0=v0)

        #--- Set Envelope
        self.a_v = np.zeros_like(self.v_grid, dtype=complex)

    #--- Class Methods
    @classmethod
    def from_TFGrid(cls, tfgrid, a_v=None):
        """
        Initialize an optical spectrum given a `TFGrid` instance.

        Use this method to create multiple `Pulse` objects with the same
        underlying coordinate grids.

        Parameters
        ----------
        tfgrid : `TFGrid` object
            An instance of a `TFGrid` object.
        a_v : array of complex, optional
            The root-power spectrum. The default (``None``) populates an empty
            spectrum.
        """
        assert isinstance(tfgrid, TFGrid), "The input must be an instance of the TFGrid class."

        #--- Copy TF Grids
        self = super().__new__(cls)
        self.__dict__.update(copy.deepcopy(tfgrid.__dict__))

        #--- Set Spectrum
        if a_v is None:
            self.a_v = np.zeros_like(self.v_grid, dtype=complex)
        else:
            assert (len(a_v)==len(self.v_grid)), "The length of a_v must match v_grid."
            self.a_v = a_v

        return self

    @classmethod
    def from_spectrum(cls, v_grid, p_v, phi_v=None, k=3, ext="zeros",
                      v_min=None, v_max=None, n_points=None, v0=None, e_p=None):
        """
        Initialize an optical spectrum by interpolating existing spectral data.

        The phase of the power spectrum, `phi_v`, must be unwrapped in order to
        obtain a valid interpolation. An additional linear phase is imposed in
        order to center the pulse at the origin of the derived time grid.

        Parameters
        ----------
        v_grid : array of float
            The frequency grid of the input data.
        p_v : array of float
            The power spectrum to be interpolated.
        phi_v : array of float, optional
            The unwrapped phase of the power spectrum. The default is ``None``,
            which initializes a transform limited pulse.
        k : int, optional
            The order of the interpolating spline, see
            `scipy.interpolate.InterpolatedUnivariateSpline`. The default is 3.
        ext : int or str, optional
            Extrapolation mode, see
            `scipy.interpolate.InterpolatedUnivariateSpline`. The default is
            "zeros".
        v_min : float, optional
            The target minimum frequency. The default (``None``) sets the
            target to the minimum of the input frequency grid.
        v_max : float, optional
            The target maximum frequency. The default (``None``) sets the
            target to the maximum of the input frequency grid.
        n_points : int, optional
            The number of grid points. The default (``None``) sets the number
            of points equal to the input frequency grid.
        v0 : float, optional
            The retarded frame reference frequency. The default (``None``) is
            the average of the resulting frequency grid.
        e_p : float, optional
            The pulse energy. The default (``None``) inherits the pulse energy
            of the input spectrum.
        """
        try:
            interpolate.InterpolatedUnivariateSpline
        except NameError:
            from scipy import interpolate
        assert (len(p_v) == len(v_grid)), "The length of p_v must match v_grid."
        v_grid = np.asarray(v_grid)
        p_v = np.asarray(p_v)
        if phi_v is not None:
            assert (len(phi_v) == len(v_grid)), "The length of phi_v must match v_grid."
            phi_v = np.asarray(phi_v)
        else:
            phi_v = np.zeros_like(v_grid)

        #--- Construct TF Grids
        if v_min is None:
            v_min =v_grid.min()
        if v_max is None:
            v_max = v_grid.max()
        if n_points is None:
            n_points = len(v_grid)
        self = super().from_freq_range(v_min, v_max, n_points, v0=v0)

        #--- Interpolate Input
        P_v_spline = interpolate.InterpolatedUnivariateSpline(
            v_grid, p_v, k=k, ext=ext)
        p_v = P_v_spline(self.v_grid)
        p_v[p_v < 0] = 0

        phi_v_spline = interpolate.InterpolatedUnivariateSpline(
            v_grid, phi_v, k=k, ext=ext)
        phi_v = phi_v_spline(self.v_grid)

        #--- Set spectrum
        self.a_v = p_v**0.5 * np.exp(1j*(phi_v - 2*pi*self.t_ref*self.v_grid))

        #--- Set Pulse Energy
        if e_p is not None:
            self.e_p = e_p

        return self

    @classmethod
    def gaussian_pulse(cls, v_min, v_max, n_points, v0, e_p, t_fwhm):
        """
        Initialize an optical power envelope with a gaussian pulse shape.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float
            The pulse's center frequency, which is also taken as the retarded
            frame reference frequency.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the optical power envelope.
        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #--- Construct TF Grids
        self = super().from_freq_range(v_min, v_max, n_points, v0=v0)

        #--- Set Spectrum
        p_t = np.exp(-np.log(16)*(self.t_grid/t_fwhm)**2)
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #--- Set Pulse Energy
        self.e_p = e_p

        return self

    @classmethod
    def sech2_pulse(cls, v_min, v_max, n_points, v0, e_p, t_fwhm):
        """
        Initialize an optical power envelope with a squared hyperbolic secant
        pulse shape.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float
            The pulse's center frequency, which is also taken as the retarded
            frame reference frequency.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the optical power envelope.
        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #--- Construct TF Grids
        self = super().from_freq_range(v_min, v_max, n_points, v0=v0)

        #--- Set Spectrum
        p_t = 1/np.cosh(2*np.arccosh(2**0.5) * self.t_grid/t_fwhm)**2
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #--- Set Pulse Energy
        self.e_p = e_p

        return self

    @classmethod
    def parabolic_pulse(cls, v_min, v_max, n_points, v0, e_p, t_fwhm):
        """
        Initialize an optical power envelope with a parabolic pulse shape.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float
            The pulse's center frequency, which is also taken as the retarded
            frame reference frequency.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the optical power envelope.
        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #--- Construct TF Grids
        self = super().from_freq_range(v_min, v_max, n_points, v0=v0)

        #--- Set Spectrum
        p_t = 1-2*(self.t_grid/t_fwhm)**2
        p_t[p_t < 0] = 0
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #--- Set Pulse Energy
        self.e_p = e_p

        return self

    @classmethod
    def lorentzian2_pulse(cls, v_min, v_max, n_points, v0, e_p, t_fwhm):
        """
        Initialize an optical power envelope with a squared lorentzian pulse
        shape.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float
            The pulse's center frequency, which is also taken as the retarded
            frame reference frequency.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the optical power envelope.
        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #--- Construct TF Grids
        self = super().from_freq_range(v_min, v_max, n_points, v0=v0)

        #--- Set Spectrum
        p_t = 1/(1+4*(2**0.5-1)*(self.t_grid/t_fwhm)**2)**2
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #--- Set Pulse Energy
        self.e_p = e_p

        return self

    #TODO: super gaussian pulse, np.exp(-np.abs(t/t_fwhm)**n)

    @classmethod
    def cw_light(cls, v_min, v_max, n_points, v0, p_avg):
        """
        Initialize an optical power spectrum with a target single frequency.

        The target frequency will be offset so that it directly aligns with one
        of the `v_grid` coordinates. The resulting frequency is taken as the
        retarded frame reference frequency.

        Parameters
        ----------
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        n_points : int
            The number of grid points.
        v0 : float
            The target continuous wave frequency.
        p_avg : float
            The average power of the CW light.
        """

        #--- Construct TF Grids
        self = super().from_freq_range(v_min, v_max, n_points)
        v0_selector = np.argmin(np.abs(self.v_grid - v0))
        v0 = self.v_grid[v0_selector]
        self.v0 = v0

        #--- Set Spectrum
        p_t = np.ones_like(self.t_grid)
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #--- Set Pulse Energy
        e_p = p_avg*self.t_window
        self.e_p = e_p

        return self

    #--- Frequency Domain Properties
    @property
    def _a_v(self):
        """
        The root-power spectrum arranged in standard fft order.

        This is a private property for the frequency domain complex envelope
        that facilitates handling fft/ifftshifts.

        Returns
        -------
        array of complex
        """
        return self.__a_v
    @_a_v.setter
    def _a_v(self, _a_v):
        assert (len(_a_v) == len(self.v_grid)), "The length of a_v must match v_grid."
        self.__a_v = _a_v

    @property
    def a_v(self):
        """
        The root-power spectrum, with units of (energy/frequency)**0.5.

        Returns
        -------
        array of complex
        """
        return fft.fftshift(self._a_v)
    @a_v.setter
    def a_v(self, a_v):
        self._a_v = fft.ifftshift(a_v)

    @property
    def _p_v(self):
        """
        The power spectrum arranged in standard fft order..

        Returns
        -------
        array of float
        """
        _a_v = self._a_v
        return _a_v.real**2 + _a_v.imag**2
    @_p_v.setter
    def _p_v(self, _p_v):
        self._a_v = _p_v**0.5 * np.exp(1j*self._phi_v)

    @property
    def p_v(self):
        """
        The power spectrum, with units of energy/frequency.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._p_v)
    @p_v.setter
    def p_v(self, p_v):
        self._p_v = fft.ifftshift(p_v)


    @property
    def _phi_v(self):
        """
        The phase of the power spectrum arranged in standard fft order.

        Returns
        -------
        array of float
        """
        return np.angle(self._a_v)
    @_phi_v.setter
    def _phi_v(self, _phi_v):
        self._a_v = self._p_v**0.5 * np.exp(1j*_phi_v)

    @property
    def phi_v(self):
        """
        The phase of the power spectrum, in radians.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._phi_v)
    @phi_v.setter
    def phi_v(self, phi_v):
        self._phi_v = fft.ifftshift(phi_v)

    @property
    def tg_v(self):
        """
        The group delay of the power spectrum, with units of time.

        Returns
        -------
        array of float
        """
        return self.t_ref - np.gradient(np.unwrap(self.phi_v)/(2*pi), self.v_grid)

    #--- Time Domain Properties
    @property
    def _a_t(self):
        """
        The complex root-power envelope arranged in standard fft order.

        This is a private property for the time domain complex envelope that
        facilitates handling fft/ifftshifts and conversion to the frequency
        domain.

        Returns
        -------
        array of complex
        """
        return fft.ifft(self._a_v*self.ndv)
    @_a_t.setter
    def _a_t(self, _a_t):
        assert (len(_a_t) == len(self.t_grid)), "The length of a_t must match t_grid."
        self._a_v = fft.fft(_a_t*self.dt)

    @property
    def a_t(self):
        """
        The complex root-power envelope, with units of (energy/time)**0.5.

        Returns
        -------
        array of complex
        """
        return fft.fftshift(self._a_t)
    @a_t.setter
    def a_t(self, a_t):
        self._a_t = fft.ifftshift(a_t)

    @property
    def _p_t(self):
        """
        The power envelope arranged in standard fft order.

        Returns
        -------
        array of float
        """
        _a_t = self._a_t
        return _a_t.real**2 + _a_t.imag**2
    @_p_t.setter
    def _p_t(self, _p_t):
        self._a_t = _p_t**0.5 * np.exp(1j*self._phi_t)

    @property
    def p_t(self):
        """
        The power envelope, with units of energy/time.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._p_t)
    @p_t.setter
    def p_t(self, p_t):
        self._p_t = fft.ifftshift(p_t)

    @property
    def _phi_t(self):
        """
        The phase of the power envelope arranged in standard fft order.

        Returns
        -------
        array of float
        """
        return np.angle(self._a_t)
    @_phi_t.setter
    def _phi_t(self, _phi_t):
        self._a_t = self._p_t**0.5 * np.exp(1j*_phi_t)

    @property
    def phi_t(self):
        """
        The phase of the power envelope, in radians.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._phi_t)
    @phi_t.setter
    def phi_t(self, phi_t):
        self._phi_t = fft.ifftshift(phi_t)

    @property
    def vg_t(self):
        """
        The instantaneous frequency of the power envelope, with units of
        frequency.

        Returns
        -------
        array of float
        """
        #TODO: replace with fft version?
        return self.v_ref + np.gradient(np.unwrap(self.phi_t)/(2*pi), self.t_grid)

    @property
    def _ra_t(self):
        """
        The real-valued instantaneous root-power arranged in standard fft
        order.

        Returns
        -------
        array of float
        """
        #--- Transform Representation
        ra_v = np.zeros_like(self.rv_grid, dtype=complex)
        ra_v[self.rn_slice] = 2**-0.5 * self.a_v
        ra_t = fft.irfft(ra_v*self.rndv, n=self.rn)
        return ra_t

    @property
    def ra_t(self):
        """
        The real-valued instantaneous root-power, with units of
        (energy/time)**0.5.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._ra_t)

    @property
    def _rp_t(self):
        """
        The instantaneous power arranged in standard fft order.

        Returns
        -------
        array of float
        """
        return self._ra_t**2

    @property
    def rp_t(self):
        """
        The instantaneous power, with units of energy/time.

        Returns
        -------
        array of float
        """
        return fft.fftshift(self._rp_t)

    #--- Energy Properties
    @property
    def e_p(self):
        """
        The pulse energy, with units of energy.

        Returns
        -------
        float
        """
        return np.sum(self.p_v*self.dv)
    @e_p.setter
    def e_p(self, e_p):
        assert (e_p > 0), "The pulse energy must be greater than 0."
        self.a_v *= (e_p/self.e_p)**0.5

    #--- Methods
    def v_width(self, n=None):
        """
        Calculate the width of the power spectrum.

        Set `n` to optionally resample the number of points and change the
        frequency resolution.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the power spectrum. The
            default (``None``) is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power spectrum.
        rms : float
            The root mean square width of the power spectrum.
        """
        results = {}

        #--- Power
        p_v = self.p_v

        #--- Resample
        if n is None:
            n = self.n
            v_grid = self.v_grid
            dv = self.dv
        else:
            assert isinstance(n, (int, np.integer)), "The number of points must be an integer."

            resampled = resample_v(self.v_grid, p_v, n)
            p_v = resampled.f_v
            v_grid = resampled.v_grid
            dv = resampled.dv

        #--- FWHM
        p_max = p_v.max()
        v_selector = v_grid[p_v >= 0.5*p_max]
        v_min = v_selector.min()
        v_max = v_selector.max()
        v_fwhm = dv + (v_max - v_min)
        results["fwhm"] = v_fwhm

        #--- RMS
        p_norm = np.sum(p_v*dv)
        v_avg = np.sum(v_grid*p_v*dv)/p_norm
        v_var = np.sum((v_grid - v_avg)**2 * p_v*dv)/p_norm
        v_rms = v_var**0.5
        results["rms"] = v_rms

        #--- Construct PowerSpectralWidths
        v_widths = collections.namedtuple("PowerSpectralWidths", results.keys())(**results)
        return v_widths

    def t_width(self, n=None):
        """
        Calculate the width of the power envelope.

        Set `n` to optionally resample the number of points and change the
        time resolution.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the power envelope. The
            default (``None``) is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power envelope.
        rms : float
            The root mean square width of the power envelope.
        """
        results = {}

        #--- Power
        p_t = self.p_t

        #--- Resample
        if n is None:
            n = self.n
            t_grid = self.t_grid
            dt = self.dt
        else:
            assert isinstance(n, (int, np.integer)), "The number of points must be an integer."

            resampled = resample_t(self.t_grid, p_t, n)
            p_t = resampled.f_t
            t_grid = resampled.t_grid
            dt = resampled.dt

        #--- FWHM
        p_max = p_t.max()
        t_selector = t_grid[p_t >= 0.5*p_max]
        t_min = t_selector.min()
        t_max = t_selector.max()
        t_fwhm = dt + (t_max - t_min)
        results["fwhm"] = t_fwhm

        #--- RMS
        p_norm = np.sum(p_t*dt)
        t_avg = np.sum(t_grid*p_t*dt)/p_norm
        t_var = np.sum((t_grid - t_avg)**2 * p_t*dt)/p_norm
        t_rms = t_var**0.5
        results["rms"] = t_rms

        #--- Construct PowerEnvelopeWidths
        t_widths = collections.namedtuple("PowerEnvelopeWidths", results.keys())(**results)
        return t_widths

    def autocorrelation(self, n=None):
        """
        Calculate the intensity autocorrelation and related diagnostic
        information.

        Set `n` to optionally resample the number of points and change the
        time resolution. The intensity autocorrelation is normalized with a max
        amplitude of 1.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the intensity
            autocorrelation. The default (``None``) is to not resample.

        Returns
        -------
        t_grid : array of float
            The time grid.
        ac_t : array of float
            The amplitude of the intensity autocorrelation.
        fwhm : float
            The full width at half maximum of the intensity
            autocorrelation.
        rms : float
            The root mean square width of the intensity autocorrelation.
        """
        results = {}

        #--- Intensity Autocorrelation
        ac_v = fft.fftshift(fft.fft(self._p_t*self.dt))**2
        ac_t = np.abs(fft.fftshift(fft.ifft(fft.ifftshift(ac_v)*self.ndv)))

        #--- Resample
        if n is None:
            n = self.n
            t_grid = self.t_grid
            dt = self.dt
        else:
            assert isinstance(n, (int, np.integer)), "The number of points must be an integer."

            resampled = resample_t(ac_t, n, dt_0=self.dt)
            ac_t = resampled.f_t
            t_grid = resampled.t_grid
            dt = resampled.dt

        ac_t /= ac_t.max()
        results["t_grid"] = t_grid
        results["ac_t"] = ac_t

        #--- FWHM
        ac_max = ac_t.max()
        t_selector = t_grid[ac_t >= 0.5*ac_max]
        t_min = t_selector.min()
        t_max = t_selector.max()
        t_fwhm = dt + (t_max - t_min)
        results["fwhm"] = t_fwhm

        #--- RMS
        ac_norm = np.sum(ac_t*dt)
        t_avg = np.sum(t_grid*ac_t*dt)/ac_norm
        t_var = np.sum((t_grid - t_avg)**2 * ac_t*dt)/ac_norm
        t_rms = t_var**0.5
        results["rms"] = t_rms

        #--- Construct Autocorrelation
        ac = collections.namedtuple("Autocorrelation", results.keys())(**results)
        return ac

    def spectrogram(self, t_fwhm=None, v_range=None, n_t=None, t_range=None):
        """
        Calculate the power spectrogram through convolution with a gaussian
        window.

        The number of points and range can be used to change the resolution in
        the time domain, but the resolution in both domains is ultimately
        limited by the time-bandwidth product of the gaussian window.

        Parameters
        ----------
        t_fwhm : float, optional
            The full width at half maximum of the gaussian window. The default
            (``None``) derives a fwhm from the bandwidth of the power spectrum.
        v_range : array_like of float, optional
            The target range of frequencies to sample. This should be given as
            (min, max) values. The default (``None``) takes the full range of
            `v_grid`.
        n_t : int or str, optional
            The number of delays to sample. Setting to "equal" gives the same
            number of delays as points in the resulting `v_grid`. The default
            (``None``) samples 4 points per fwhm of the gaussian window.
        t_range : array_like of float, optional
            The range of delays to sample. This should be given as (min, max)
            values. The default (``None``) takes the full range of the
            resulting `t_grid`.

        Returns
        -------
        v_grid : array of float
            The frequency grid
        t_grid : array of float
            The time grid.
        spgr : array of float
            The amplitude of the spectrogram. The first axis corresponds to
            frequency and the second axis to time.
        extent : tuple of float
            A bounding box suitable for use with `matplotlib`'s `imshow`
            function with the `origin` keyword set to "lower". This
            reliably centers the pixels on the `t_grid` and `v_grid` grids.

        Notes
        -----
        The full width at half maximum of the gaussian window should be similar
        to the full width at half maximum of the pulse in order to evenly
        distribute resolution between the time and frequency domains.
        """
        results = {}

        #--- Resample
        if v_range is None:
            n = self.n
            v_grid = self.v_grid
            ndv = self.ndv
            a_t = self.a_t
            t_grid = self.t_grid
            dt = self.dt
        else:
            v_range = np.asarray(v_range)
            assert (v_range.min() >= self.v_grid.min()), "The requested minimum frequency cannot be less than the minimum possible frequency."
            assert (v_range.max() <= self.v_grid.max()), "The requested maximum frequency cannot be greater than the maximum possible frequency."

            v_min_selector = np.argmin(np.abs(self.v_grid - v_range.min()))
            v_max_selector = np.argmin(np.abs(self.v_grid - v_range.max()))
            v_grid = self.v_grid[v_min_selector:v_max_selector+1]
            n = len(v_grid)
            ndv = n*self.dv

            a_v = self.a_v[v_min_selector:v_max_selector+1]
            a_t = fft.ifft(fft.ifftshift(a_v)*ndv)
            dt = 1/(n*self.dv)
            t_grid = dt*np.arange(n)
            t_grid -= fft.ifftshift(t_grid)[0]

        results["v_grid"] = v_grid

        #--- Set Gate
        if t_fwhm is None:
            v_rms = self.v_width().rms
            t_fwhm = np.log(4)**0.5 / (2*pi*v_rms)

        g_t = np.exp(-np.log(16)*(t_grid/t_fwhm)**2)**0.5

        g_t /= np.sum(np.abs(g_t)**2 * dt)**0.5
        g_v = fft.fftshift(fft.fft(g_t*dt))

        #--- Set Delays
        if t_range is None:
            t_min, t_max = t_grid.min(), t_grid.max()
        else:
            t_range = np.asarray(t_range)
            assert (t_range.min() >= t_grid.min()), "The requested minimum delay cannot be less than the minimum possible delay {:.2f}.".format(t_grid.min())
            assert (t_range.max() <= t_grid.max()), "The requested maximum delay cannot be greater than the maximum possible delay {:.2f}.".format(t_grid.max())

            t_min, t_max = t_range

        if n_t is None:
            n_t = int(4*round((t_max - t_min)/t_fwhm))
        elif isinstance(n_t, str):
            assert (n_t in ["equal"]), "'{:}' is not part of the valid string arguments for n_t".format(n_t)

            n_t = n
        else:
            assert isinstance(n_t, (int, np.integer)), "The number of points must be an integer."
            assert (n_t > 1), "The number of points must be greater than 1."

        delay_t_grid = np.linspace(t_min, t_max, n_t)
        delay_dt = (t_max - t_min)/(n_t - 1)
        results["t_grid"] = delay_t_grid

        gate_pulses_v = g_v[:, np.newaxis] * np.exp(1j*2*pi*delay_t_grid[np.newaxis, :]*v_grid[:, np.newaxis])
        gate_pulses_t = fft.ifft(fft.ifftshift(gate_pulses_v, axes=0)*ndv, axis=0)

        #--- Spectrogram
        spg_t = a_t[:, np.newaxis] * gate_pulses_t
        spg_v = fft.fftshift(fft.fft(spg_t*dt, axis=0), axes=0) #!!!

        p_spg = spg_v.real**2 + spg_v.imag**2
        results["spg"] = p_spg

        #--- Extent
        extent = (delay_t_grid.min()-0.5*delay_dt, delay_t_grid.max()+0.5*delay_dt,
                  v_grid.min()-0.5*self.dv, v_grid.max()+0.5*self.dv)
        results["extent"] = extent

        #--- Construct Spectrogram
        spg = collections.namedtuple("Spectrogram", results.keys())(**results)
        return spg

