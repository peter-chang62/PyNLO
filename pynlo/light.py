# -*- coding: utf-8 -*-
"""
Optical pulses in the time and frequency domains.

Notes
-----
The public facing routines and properties of the defined classes have inputs
and outputs that are arranged such that the coordinate arrays are monotonically
ordered. Many of the associated private methods and properties are arranged in
the standard fft order.

"""

__all__ = ["Pulse"]


# %% Imports

import collections

import numpy as np
from scipy.constants import pi

from pynlo.utility import TFGrid, fft, resample_v, resample_t
from pynlo.utility.misc import ArrayProperty, replace


# %% Collections

PowerSpectralWidths = collections.namedtuple("PowerSpectralWidths", ["fwhm", "rms", "eff"])

PowerEnvelopeWidths = collections.namedtuple("PowerEnvelopeWidths", ["fwhm", "rms", "eff"])

Autocorrelation = collections.namedtuple("Autocorrelation", ["t_grid", "ac_t", "fwhm", "rms", "eff"])

Spectrogram = collections.namedtuple("Spectrogram", ["v_grid", "t_grid", "spg", "extent"])


# %% Classes

class Pulse(TFGrid):
    """
    An optical pulse, defined over complementary time and frequency grids.

    The default initializer creates an empty `Pulse` object. Set one of four
    attributes, `a_v`, `p_v`, `a_t`, or `p_t`, to populate the pulse spectrum
    or temporal envelope.

    Parameters
    ----------
    n_points : int
        The number of grid points.
    dv : float
        The frequency step size. This is equal to the reciprocal of the time
        window.
    v_ref : float
        The target central frequency of the grid.
    a_v : array_like of complex, optional
        The root-power spectrum. The default selection is an empty spectrum.
    alias : int, optional
        The number of harmonics supported by the real-valued time grid without
        aliasing. The default is 1, the fundamental harmonic. A higher number
        may be useful when simulating nonlinear interactions.

    Notes
    -----
    See :py:class:`~pynlo.utility.TFGrid` for other attributes.

    The power spectrum and power envelope are normalized to the pulse energy
    `e_p`::

        e_p == np.sum(p_v*dv) == np.sum(p_t*dt) == np.sum(rp_t*rdt)

    The amplitude of the root-power spectrum of the complex envelope is a
    factor of ``2**0.5`` larger than the root-power spectrum of the
    real-valued representation. This difference is due to the real DFT being
    only the positive side of a double-sided spectrum. When transforming
    between the two representations use the following normalization::

        a_v = 2**0.5 * ra_v[rn_slice]
        ra_v[rn_slice] = 2**-0.5 * a_v

    """

    def __init__(self, n_points, dv, v_ref, a_v=None, alias=1):
        """
        Initialize a pulse given the total number of points, the frequency step
        size (reciprocal of the time window), and a target central frequency.

        If necessary, the reference frequency will be increased so that the
        grid is formed without any negative frequencies.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        dv : float
            The frequency step size. This is equal to the reciprocal of the
            time window.
        v_ref : float
            The target central frequency of the grid.
        a_v : array_like of complex, optional
            The root-power spectrum. The default selection is an empty
            spectrum.
        alias : int, optional
            The number of harmonics supported by the real-valued time grid
            without aliasing. The default is 1, the fundamental harmonic. A
            higher number may be useful when simulating nonlinear interactions.

        """
        #---- Initialize Grids
        super().__init__(n_points, dv, v_ref, alias=alias)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        if a_v is None:
            self.a_v = 0
        else:
            assert (len(a_v)==len(self.v_grid)), "The length of a_v must match v_grid."
            self.a_v = np.asarray(a_v, dtype=complex)

    #---- Class Methods
    @classmethod
    def FromTFGrid(cls, tfgrid, a_v=None):
        """
        Initialize a pulse given a `TFGrid` instance.

        Use this method to create `Pulse` objects from an existing set of
        coordinate grids.

        Parameters
        ----------
        tfgrid : pynlo.utility.TFGrid
            An instance of a `TFGrid` object.
        a_v : array_like of complex, optional
            The root-power spectrum. The default populates an empty spectrum.

        """
        assert isinstance(tfgrid, TFGrid), "The input must be an instance of the TFGrid class."

        #---- Initialize Grids
        self = super().__new__(cls)
        self.__dict__.update(tfgrid.__dict__) # copy TFGrid
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        if a_v is None:
            self.a_v = 0
        else:
            assert (len(a_v)==len(self.v_grid)), "The length of a_v must match v_grid."
            self.a_v = a_v
        return self

    @classmethod
    def FromPowerSpectrum(cls, p_v, n_points, v_min, v_max, v0=None, e_p=None, phi_v=None, **kwargs):
        """
        Initialize a pulse using existing spectral data.

        Parameters
        ----------
        p_v : callable -> array_like of float
            The power spectrum to be evaluated along the resulting frequency
            grid.
        n_points : int, optional
            The number of grid points.
        v_min : float, optional
            The target minimum frequency.
        v_max : float, optional
            The target maximum frequency.
        v0 : float, optional
            The comoving frame reference frequency. The default selection is
            the center of the resulting frequency grid.
        e_p : float, optional
            The pulse energy. The default inherits the pulse energy of the
            input spectrum.
        phi_v : callable -> array_like of float, optional
            The phase of the power spectrum. The default initializes a
            transform limited pulse.
        """
        assert callable(p_v), "The power spectrum must be a callable function."

        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Evaluate Input
        p_v = np.asarray(p_v(self.v_grid), dtype=float)
        p_v[p_v < 0] = 0

        if phi_v is not None:
            assert callable(phi_v), "The phase must be a callable function."
            phi_v = np.asarray(phi_v(self.v_grid), dtype=float)
        else:
            phi_v = np.zeros_like(self.v_grid)

        #---- Set spectrum
        self.a_v = p_v**0.5 * np.exp(1j*phi_v)

        #---- Set Pulse Energy
        if e_p is not None:
            self.e_p = e_p
        return self

    @classmethod
    def Gaussian(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, m=1, **kwargs):
        """
        Initialize a gaussian or super-gaussian pulse.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float
            The pulse's center frequency. Also taken as the reference frequency
            for the comoving frame.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the pulse's power envelope.
        m : float, optional
            The super-gaussian order. Default is 1.

        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        p_t = 2**(-((2*self.t_grid/t_fwhm)**2)**m)
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #---- Set Pulse Energy
        self.e_p = e_p
        return self

    @classmethod
    def Sech(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, **kwargs):
        """
        Initialize a squared hyperbolic secant pulse.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float
            The pulse's center frequency. Also taken as the reference frequency
            for the comoving frame.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the pulse's power envelope.

        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        t0 = t_fwhm / (2 * np.arccosh(2**0.5))
        p_t = (1/np.cosh(self.t_grid/t0))**2
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #---- Set Pulse Energy
        self.e_p = e_p
        return self

    @classmethod
    def Parabolic(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, **kwargs):
        """
        Initialize a parabolic pulse.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float
            The pulse's center frequency. Also taken as the reference frequency
            for the comoving frame.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the pulse's power envelope.

        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        p_t = 1-2*(self.t_grid/t_fwhm)**2
        p_t[p_t < 0] = 0
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #---- Set Pulse Energy
        self.e_p = e_p
        return self

    @classmethod
    def Lorentzian(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, **kwargs):
        """
        Initialize a squared Lorentzian pulse.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float
            The pulse's center frequency. Also taken as the reference frequency
            for the comoving frame.
        e_p : float
            The pulse energy.
        t_fwhm : float
            The full width at half maximum of the pulse's power envelope.

        """
        assert (t_fwhm > 0), "The pulse width must be greater than 0."

        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        p_t = 1/(1+4*(2**0.5-1)*(self.t_grid/t_fwhm)**2)**2
        phi_t = 2*pi*(v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #---- Set Pulse Energy
        self.e_p = e_p
        return self

    @classmethod
    def CW(cls, n_points, v_min, v_max, v0, p_avg, **kwargs):
        """
        Initialize an optical spectrum with a single target frequency.

        The target frequency will be offset so that it directly aligns with one
        of the `v_grid` coordinates.

        Parameters
        ----------
        n_points : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.
        v0 : float
            The target continuous wave frequency. Also taken as the reference
            frequency for the comoving frame.
        p_avg : float
            The average power of the CW light.

        """
        #---- Initialize Grids
        self = super().FromFreqRange(n_points, v_min, v_max, v0=v0, **kwargs)
        self.__a_v = np.empty_like(self.v_grid, dtype=complex)

        #---- Set Spectrum
        p_t = np.ones_like(self.t_grid)
        phi_t = 2*pi*(self.v0-self.v_ref)*self.t_grid
        self.a_t = p_t**0.5 * np.exp(1j*phi_t)

        #---- Set Pulse Energy
        e_p = p_avg*self.t_window
        self.e_p = e_p
        return self

    #---- Frequency Domain Properties
    @property
    def a_v(self):
        """
        The root-power spectrum, with units of ``(J/Hz)**0.5``.

        Returns
        -------
        ndarray of complex

        """
        return self.__a_v
    @a_v.setter
    def a_v(self, a_v):
        self.__a_v[...] = a_v

    @ArrayProperty
    def _a_v(self, key=...):
        """
        The root-power spectrum arranged in standard fft order.

        Returns
        -------
        ndarray of complex

        """
        return fft.ifftshift(self.a_v)[key]
    @_a_v.setter
    def _a_v(self, _a_v, key=...):
        if key is not ...:
            _a_v = replace(self._a_v, _a_v, key)
        self.a_v = fft.fftshift(_a_v)

    @ArrayProperty
    def p_v(self, key=...):
        """
        The power spectrum, with units of ``J/Hz``.

        Returns
        -------
        ndarray of float

        """
        return self.a_v[key].real**2 + self.a_v[key].imag**2
    @p_v.setter
    def p_v(self, p_v, key=...):
        self.a_v[key] = p_v**0.5 * np.exp(1j*self.phi_v[key])

    @ArrayProperty
    def _p_v(self, key=...):
        """
        The power spectrum arranged in standard fft order.

        Returns
        -------
        ndarray of float

        """
        return fft.ifftshift(self.p_v)[key]
    @_p_v.setter
    def _p_v(self, _p_v, key=...):
        if key is not ...:
            _p_v = replace(self._p_v, _p_v, key)
        self.p_v = fft.fftshift(_p_v)

    @ArrayProperty
    def phi_v(self, key=...):
        """
        The phase of the power spectrum, in radians.

        Returns
        -------
        ndarray of float

        """
        return np.angle(self.a_v[key])
    @phi_v.setter
    def phi_v(self, phi_v, key=...):
        self.a_v[key] = self.p_v[key]**0.5 * np.exp(1j*phi_v)

    @ArrayProperty
    def _phi_v(self, key=...):
        """
        The phase of the power spectrum arranged in standard fft order.

        Returns
        -------
        ndarray of float

        """
        return fft.ifftshift(self.phi_v)[key]
    @_phi_v.setter
    def _phi_v(self, _phi_v, key=...):
        if key is not ...:
            _phi_v = replace(self._phi_v, _phi_v, key)
        self.phi_v = fft.fftshift(_phi_v)

    @ArrayProperty
    def tg_v(self, key=...):
        """
        The group delay of the power spectrum, with units of ``s``.

        Returns
        -------
        ndarray of float

        """
        return self.t_ref - np.gradient(np.unwrap(self.phi_v)/(2*pi), self.v_grid, edge_order=2)[key]

    def v_width(self, n=None):
        """
        Calculate the spectral width of the power spectrum.

        Set `n` to optionally resample the number of points and change the
        frequency resolution.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the power spectrum. The
            default is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power spectrum.
        rms : float
            The root mean square width of the power spectrum.
        eff : float
            The effective width of the power spectrum.

        """
        #---- Power
        p_v = self.p_v

        #---- Resample
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

        #---- FWHM
        p_max = p_v.max()
        v_selector = v_grid[p_v >= 0.5*p_max]
        v_fwhm = dv + (v_selector.max() - v_selector.min())

        #---- RMS
        p_norm = np.sum(p_v*dv)
        v_avg = np.sum(v_grid*p_v*dv)/p_norm
        v_var = np.sum((v_grid - v_avg)**2 * p_v*dv)/p_norm
        v_rms = v_var**0.5

        #---- Effective
        v_eff = 1/np.sum((p_v/p_norm)**2 * dv)

        #---- Construct PowerSpectralWidths
        v_widths = PowerSpectralWidths(fwhm=v_fwhm, rms=v_rms, eff=v_eff)
        return v_widths

    #---- Time Domain Properties
    @ArrayProperty
    def a_t(self, key=...):
        """
        The complex root-power envelope, with units of ``(J/s)**0.5``.

        Returns
        -------
        ndarray of complex

        """
        return fft.fftshift(self._a_t)[key]
    @a_t.setter
    def a_t(self, a_t, key=...):
        if key is not ...:
            a_t = replace(self.a_t, a_t, key)
        self._a_t = fft.ifftshift(a_t)

    @ArrayProperty
    def _a_t(self, key=...):
        """
        The complex root-power envelope arranged in standard fft order.

        Returns
        -------
        ndarray of complex

        """
        return fft.ifft(self._a_v, fsc=self.dt)[key]
    @_a_t.setter
    def _a_t(self, _a_t, key=...):
        if key is not ...:
            _a_t = replace(self._a_t, _a_t, key)
        self._a_v = fft.fft(_a_t, fsc=self.dt)

    @ArrayProperty
    def p_t(self, key=...):
        """
        The power envelope, with units of ``J/s``.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._p_t)[key]
    @p_t.setter
    def p_t(self, p_t, key=...):
        if key is not ...:
            p_t = replace(self.p_t, p_t, key)
        self._p_t = fft.ifftshift(p_t)

    @ArrayProperty
    def _p_t(self, key=...):
        """
        The power envelope arranged in standard fft order.

        Returns
        -------
        ndarray of float

        """
        _a_t = self._a_t[key]
        return _a_t.real**2 + _a_t.imag**2
    @_p_t.setter
    def _p_t(self, _p_t, key=...):
        self._a_t[key] = _p_t**0.5 * np.exp(1j*self._phi_t[key])

    @ArrayProperty
    def phi_t(self, key=...):
        """
        The phase of the power envelope, in radians.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._phi_t)[key]
    @phi_t.setter
    def phi_t(self, phi_t, key=...):
        if key is not ...:
            phi_t = replace(self.phi_t, phi_t, key)
        self._phi_t = fft.ifftshift(phi_t)

    @ArrayProperty
    def _phi_t(self, key=...):
        """
        The phase of the power envelope arranged in standard fft order.

        Returns
        -------
        ndarray of float

        """
        return np.angle(self._a_t[key])
    @_phi_t.setter
    def _phi_t(self, _phi_t, key=...):
        self._a_t[key] = self._p_t[key]**0.5 * np.exp(1j*_phi_t)

    @ArrayProperty
    def vg_t(self, key=...):
        """
        The instantaneous frequency of the power envelope, with units of
        ``Hz``.

        Returns
        -------
        ndarray of float

        """
        return self.v_ref + np.gradient(np.unwrap(self.phi_t)/(2*pi), self.t_grid, edge_order=2)[key]

    @ArrayProperty
    def ra_t(self, key=...):
        """
        The real-valued instantaneous root-power, with units of
        ``(J/s)**0.5``.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._ra_t)[key]

    @ArrayProperty
    def _ra_t(self, key=...):
        """
        The real-valued instantaneous root-power arranged in standard fft
        order.

        Returns
        -------
        ndarray of float

        """
        ra_v = np.zeros_like(self.rv_grid, dtype=complex)
        ra_v[self.rn_slice] = 2**-0.5 * self.a_v
        ra_t = fft.irfft(ra_v, fsc=self.rdt, n=self.rn)
        return ra_t[key]

    @ArrayProperty
    def rp_t(self, key=...):
        """
        The instantaneous power, with units of ``J/s``.

        Returns
        -------
        ndarray of float

        """
        return fft.fftshift(self._rp_t)[key]

    @ArrayProperty
    def _rp_t(self, key=...):
        """
        The instantaneous power arranged in standard fft order.

        Returns
        -------
        ndarray of float

        """
        return self._ra_t[key]**2

    def t_width(self, n=None):
        """
        Calculate the temporal width of the power envelope.

        Set `n` to optionally resample the number of points and change the
        time resolution.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the power envelope. The
            default is to not resample.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the power envelope.
        rms : float
            The root mean square width of the power envelope.
        eff : float
            The effective width of the power envelope.

        """
        #---- Power
        p_t = self.p_t

        #---- Resample
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

        #---- FWHM
        p_max = p_t.max()
        t_selector = t_grid[p_t >= 0.5*p_max]
        t_fwhm = dt + (t_selector.max() - t_selector.min())

        #---- RMS
        p_norm = np.sum(p_t*dt)
        t_avg = np.sum(t_grid*p_t*dt)/p_norm
        t_var = np.sum((t_grid - t_avg)**2 * p_t*dt)/p_norm
        t_rms = t_var**0.5

        #---- Effective
        t_eff = 1/np.sum((p_t/p_norm)**2 * dt)

        #---- Construct PowerEnvelopeWidths
        t_widths = PowerEnvelopeWidths(fwhm=t_fwhm, rms=t_rms, eff=t_eff)
        return t_widths

    #---- Energy Properties
    @property
    def e_p(self):
        """
        The pulse energy, with units of ``J``.

        Returns
        -------
        float

        """
        return np.sum(self.p_v*self.dv)
    @e_p.setter
    def e_p(self, e_p):
        assert (e_p > 0), "The pulse energy must be greater than 0."
        self.a_v *= (e_p/self.e_p)**0.5

    #---- Indirect Measures
    def autocorrelation(self, n=None):
        """
        Calculate the intensity autocorrelation and related diagnostic
        information.

        Set `n` to optionally resample the number of points and change the
        time resolution. The intensity autocorrelation is normalized with a
        max amplitude of 1.

        Parameters
        ----------
        n : int, optional
            The number of points at which to resample the intensity
            autocorrelation. The default is to not resample.

        Returns
        -------
        t_grid : ndarray of float
            The time grid.
        ac_t : ndarray of float
            The amplitude of the intensity autocorrelation.
        fwhm : float
            The full width at half maximum of the intensity autocorrelation.
        rms : float
            The root mean square width of the intensity autocorrelation.
        eff : float
            The effective width of the intensity autocorrelation.

        """
        #---- Intensity Autocorrelation
        ac_v = np.abs(fft.fftshift(fft.fft(self._p_t, fsc=self.dt)))**2
        ac_t = fft.fftshift(fft.ifft(fft.ifftshift(ac_v), fsc=self.dt, overwrite_x=True).real)

        #---- Resample
        if n is None:
            n = self.n
            t_grid = self.t_grid
            dt = self.dt
        else:
            assert isinstance(n, (int, np.integer)), "The number of points must be an integer."
            resampled = resample_t(self.t_grid, ac_t, n)
            ac_t = resampled.f_t
            t_grid = resampled.t_grid
            dt = resampled.dt
        ac_t /= ac_t.max()

        #---- FWHM
        ac_max = ac_t.max()
        t_selector = t_grid[ac_t >= 0.5*ac_max]
        t_fwhm = dt + (t_selector.max() - t_selector.min())

        #---- RMS
        ac_norm = np.sum(ac_t*dt)
        t_avg = np.sum(t_grid*ac_t*dt)/ac_norm
        t_var = np.sum((t_grid - t_avg)**2 * ac_t*dt)/ac_norm
        t_rms = t_var**0.5

        #---- Effective
        t_eff = 1/np.sum((ac_t/ac_norm)**2 * dt)

        #---- Construct Autocorrelation
        ac = Autocorrelation(t_grid=t_grid, ac_t=ac_t, fwhm=t_fwhm, rms=t_rms, eff=t_eff)
        return ac

    def spectrogram(self, t_fwhm=None, v_range=None, n_t=None, t_range=None):
        """
        Calculate the power spectrogram of the pulse through convolution with a
        gaussian window.

        Parameters
        ----------
        t_fwhm : float, optional
            The full width at half maximum of the gaussian window. The default
            derives a fwhm from the bandwidth of the power spectrum.
        v_range : array_like of float, optional
            The target range of frequencies to sample. This should be given as
            (min, max) values. The default takes the full range of `v_grid`.
        n_t : int or str, optional
            The number of delays to sample. Setting to "equal" gives the same
            number of delays as points in the resulting `v_grid`. The default
            samples 4 points per fwhm of the gaussian window.
        t_range : array_like of float, optional
            The range of delays to sample. This should be given as (min, max)
            values. The default takes the full range of the resulting
            `t_grid`.

        Returns
        -------
        v_grid : ndarray of float
            The frequency grid.
        t_grid : ndarray of float
            The time grid.
        spg : ndarray of float
            The amplitude of the spectrogram. The first axis corresponds to
            frequency and the second axis to time.
        extent : tuple of float
            A bounding box suitable for use with `matplotlib`'s `imshow`
            function with the `origin` keyword set to "lower". This
            reliably centers the pixels on the `v_grid` and `t_grid`
            coordinates.

        Notes
        -----
        The resolution in both domains is limited by the time-bandwidth product
        of the gaussian window. The full width at half maximum of the gaussian
        window should be similar to the full width at half maximum of the pulse
        in order to evenly distribute resolution bandwidth between the time and
        frequency domains.

        """
        #---- Resample
        if v_range is None:
            n = self.n
            v_grid = self.v_grid
            a_t = self.a_t
            t_grid = self.t_grid
            dt = self.dt
        else:
            v_range = np.asarray(v_range)
            assert (v_range.min() >= self.v_grid.min()), (
                "The requested minimum frequency cannot be less than the"
                " minimum possible frequency.")
            assert (v_range.max() <= self.v_grid.max()), (
                "The requested maximum frequency cannot be greater than the"
                " maximum possible frequency.")

            v_min_selector = np.argmin(np.abs(self.v_grid - v_range.min()))
            v_max_selector = np.argmin(np.abs(self.v_grid - v_range.max()))
            v_grid = self.v_grid[v_min_selector:v_max_selector+1]
            n = len(v_grid)
            dt = 1/(n*self.dv)

            a_v = self.a_v[v_min_selector:v_max_selector+1]
            a_t = fft.fftshift(fft.ifft(fft.ifftshift(a_v), fsc=dt, overwrite_x=True))
            t_grid = dt*(np.arange(n) - (n//2))

        #---- Set Gate
        if t_fwhm is None:
            v_rms = self.v_width().rms
            t_fwhm = np.log(4)**0.5 / (2*pi*v_rms)

        g_t = (2**(-(2*t_grid/t_fwhm)**2))**0.5

        g_t /= np.sum(np.abs(g_t)**2 * dt)**0.5
        g_v = fft.fftshift(fft.fft(fft.ifftshift(g_t), fsc=dt))

        #---- Set Delays
        if t_range is None:
            t_min, t_max = t_grid.min(), t_grid.max()
        else:
            t_range = np.asarray(t_range)
            assert (t_range.min() >= t_grid.min()), (
                "The requested minimum delay cannot be less than the minimum"
                " possible delay {:.2f}.".format(t_grid.min()))
            assert (t_range.max() <= t_grid.max()), (
                "The requested maximum delay cannot be greater than the"
                " maximum possible delay {:.2f}.".format(t_grid.max()))
            t_min, t_max = t_range

        if n_t is None:
            n_t = int(4*round((t_max - t_min)/t_fwhm))
        elif isinstance(n_t, str):
            assert (n_t in ["equal"]), ("'{:}' is not part of the valid string"
                                        " arguments for n_t".format(n_t))
            n_t = n
        else:
            assert isinstance(n_t, (int, np.integer)), "The number of points must be an integer."
            assert (n_t > 1), "The number of points must be greater than 1."
        delay_t_grid = np.linspace(t_min, t_max, n_t)
        delay_dt = (t_max - t_min)/(n_t - 1)

        gate_pulses_v = (g_v[:, np.newaxis]
                         * np.exp(-1j*2*pi*delay_t_grid[np.newaxis, :]*v_grid[:, np.newaxis]))
        gate_pulses_t = fft.fftshift(fft.ifft(
            fft.ifftshift(gate_pulses_v, axis=0), fsc=dt, axis=0, overwrite_x=True), axis=0)

        #---- Spectrogram
        spg_t = a_t[:, np.newaxis] * gate_pulses_t
        spg_v = fft.fftshift(fft.fft(
            fft.ifftshift(spg_t, axis=0), fsc=dt, axis=0, overwrite_x=True), axis=0)
        p_spg = spg_v.real**2 + spg_v.imag**2

        #---- Extent
        extent = (delay_t_grid.min()-0.5*delay_dt, delay_t_grid.max()+0.5*delay_dt,
                  v_grid.min()-0.5*self.dv, v_grid.max()+0.5*self.dv)

        #---- Construct Spectrogram
        spg = Spectrogram(v_grid=v_grid, t_grid=delay_t_grid, spg=p_spg, extent=extent)
        return spg
