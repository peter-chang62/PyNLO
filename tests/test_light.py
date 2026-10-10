# -*- coding: utf-8 -*-
"""
TODO: module docs... testing for pynlo.light methods and classes
"""

# %% Imports

import numpy as np
from scipy import constants
if __name__ == "__main__":
    import matplotlib.pyplot as plt

from pynlo import light
from pynlo.utility import fft


# %% Constants

pi = constants.pi


# %% Pulse

def test_pulse_shapes():
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15
    min_time_window = 0.5e-12

    #--- Gaussian Pulse
    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm,
                            min_time_window=min_time_window)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dt, rtol=0)

    #--- Sech**2 Pulse
    test = light.Pulse.Sech(n, v_min, v_max, v_0, e_p, t_fwhm,
                            min_time_window=min_time_window)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dt, rtol=0)

    #--- Parabolic Pulse
    test = light.Pulse.Parabolic(n, v_min, v_max, v_0, e_p, t_fwhm,
                            min_time_window=min_time_window)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dt, rtol=0)

    #--- Lorentzian**2 Pulse
    test = light.Pulse.Lorentzian(n, v_min, v_max, v_0, e_p, t_fwhm,
                            min_time_window=min_time_window)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dt, rtol=0)

    #--- CW Light
    p_p = 1
    test = light.Pulse.CW(n, v_min, v_max, v_0, p_p, min_time_window=min_time_window)
    assert np.isclose(p_p, test.e_p/test.t_window)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)

def test_pulse_properties():
    """
    Test Pulse properties on a generated gaussian pulse.
    """
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15
    min_time_window = 0.5e-12
    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm,
                                min_time_window=min_time_window)

    #--- Frequency Domain Properties
    assert all(test.a_v == fft.fftshift(test._a_v))
    assert all(test.p_v == fft.fftshift(test._p_v))
    assert all(test.phi_v == fft.fftshift(test._phi_v))
    assert np.allclose(np.log10(test.p_v**0.5 * np.exp(1j*test.phi_v)), np.log10(test.a_v), equal_nan=True)

    #--- Time Domain Properties
    assert all(test.a_t == fft.fftshift(test._a_t))
    assert all(test.p_t == fft.fftshift(test._p_t))
    assert all(test.phi_t == fft.fftshift(test._phi_t))
    assert np.allclose(np.log10(test.p_t**0.5 * np.exp(1j*test.phi_t)), np.log10(test.a_t), equal_nan=True)
    assert all(test.ra_t == fft.fftshift(test._ra_t))
    assert all(test.rp_t == fft.fftshift(test._rp_t))
    assert np.allclose(np.log10(test.rp_t), np.log10(test.ra_t**2), equal_nan=True)

    #--- Methods
    v_w = test.v_width()
    t_w = test.t_width()
    v_sigma = 0.5 * v_w.rms  # v_width().rms is the full (2σ) rms width
    assert np.isclose(np.log(4)**0.5 / (2*pi*v_sigma), t_w.fwhm, atol=test.dt)
    ac = test.autocorrelation()
    assert np.isclose(ac.rms, 2**0.5 * t_w.rms, atol=0)

def test_min_time_window():
    v_min, v_max, v_0, e_p, t_fwhm = 100e12, 500e12, 300e12, 1, 100e-15
    # n too small: grid is enlarged so the time window is at least min_time_window
    for min_time_window in (0.5e-12, 1e-12):
        p = light.Pulse.Gaussian(2, v_min, v_max, v_0, e_p, t_fwhm,
                                 min_time_window=min_time_window)
        assert p.t_window >= min_time_window
        # endpoints are targets (TFGrid snaps v_ref to a multiple of dv); the span is exact
        assert np.isclose(p.v_grid[-1] - p.v_grid[0], v_max - v_min)
    # n already sufficient: user's n is kept
    p = light.Pulse.Gaussian(256, v_min, v_max, v_0, e_p, t_fwhm, min_time_window=0.5e-12)
    assert p.n == 256

def test_width_rms_is_full_width():
    """
    t_width().rms is the full (2σ) rms width, not σ (convention since 84452fd).
    """
    v_min, v_max, v_0, e_p, t_fwhm = 100e12, 500e12, 300e12, 1, 100e-15
    test = light.Pulse.Gaussian(2**8, v_min, v_max, v_0, e_p, t_fwhm,
                                min_time_window=0.5e-12)
    sigma_t = t_fwhm / (2*(2*np.log(2))**0.5)
    assert np.isclose(test.t_width().rms, 2*sigma_t, rtol=1e-3)

#--- Exploratory
if __name__ == "__main__":
    test_pulse_shapes()
    test_pulse_properties()

    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 40e-15
    min_time_window = 0.5e-12
    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm,
                                min_time_window=min_time_window)

    # test.phi_v += 2*pi*(.1*1e-12)*test.v_grid

    #--- Complex Envelope
    plt.figure("Complex Envelope")
    plt.clf()
    ax0 = plt.subplot2grid((2,1), (0,0))
    ax1 = plt.subplot2grid((2,1), (1,0), sharex=ax0)
    ax0.semilogy(1e12*test.t_grid, test.p_t)
    ax1.plot(1e12*test.t_grid, 1e-12*test.vg_t)
    plt.tight_layout()

    #--- Power Spectrum
    plt.figure("Power Spectrum")
    plt.clf()
    ax0 = plt.subplot2grid((2,1), (0,0))
    ax1 = plt.subplot2grid((2,1), (1,0), sharex=ax0)
    ax0.semilogy(1e-12*test.v_grid, test.p_v)
    ax1.plot(1e-12*test.v_grid, 1e12*test.tg_v)
    plt.tight_layout()

    #--- Autocorrelation
    ac = test.autocorrelation()

    plt.figure("Autocorrelation")
    plt.clf()
    plt.plot(ac.t_grid, ac.ac_t)

    #--- Spectrogram
    spg = test.spectrogram()
    plt.figure("Spectrogram")
    plt.clf()
    plt.imshow(10*np.log10(spg.spg), origin="lower", extent=spg.extent, aspect="auto")
    plt.tight_layout()
