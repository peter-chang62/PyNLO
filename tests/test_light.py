# -*- coding: utf-8 -*-
"""
TODO: module docs... testing for pynlo.light methods and classes
"""

# %% Imports

import numpy as np
from scipy import constants, integrate, fft
if __name__ == "__main__":
    import matplotlib.pyplot as plt

from pynlo import light

# %% Constants

pi = constants.pi

# %% Derivatives

class TestDerivativeT():
    def test_all(self):
        if __name__ == "__main__":
            self.test_r1()
            self.test_r2()
            self.test_r3()
            self.test_r4()

            self.test_c1()
            self.test_c2()
            self.test_c3()
            self.test_c4()

    #--- Real-Valued Representation Input, Even Number of Points
    def test_r1(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1), -1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1), 1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Real-Valued Representation Input, Odd Number of Points
    def test_r2(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1), -1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1), 1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Real-Valued Representation Input, Even Number of Points, Frequency Offset
    def test_r3(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5
        v_ref = 100.

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1, v_ref=v_ref), -1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1, v_ref=v_ref), 1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Real-Valued Representation Input, Odd Number of Points, Frequency Offset
    def test_r4(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5
        v_ref = 100.

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1, v_ref=v_ref), -1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1, v_ref=v_ref), 1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1), -1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1), 1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1), -1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1), 1)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Complex Envelope Input, Even Number of Points, Frequency Offset
    def test_c3(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5
        v_ref = 100.

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1, v_ref=v_ref), -1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1, v_ref=v_ref), 1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

    #--- Complex Envelope Input, Odd Number of Points, Frequency Offset
    def test_c4(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5
        v_ref = 100.

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
        a_t -= a_t.mean()
        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, 1, v_ref=v_ref), -1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

        a_t_ident = light.derivative_t(t_grid, light.derivative_t(t_grid, a_t, -1, v_ref=v_ref), 1, v_ref=v_ref)
        assert np.allclose(a_t_ident, a_t, equal_nan=True)

class TestDerivativeV():
    def test_all(self):
        if __name__ == "__main__":
            self.test_c1()
            self.test_c2()
            self.test_c3()
            self.test_c4()

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        n_2 = n//2
        dv = 0.5

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j
        a_v -= a_v.mean()
        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, 1), -1)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, -1), 1)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        n_2 = n//2
        dv = 0.5

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j
        a_v -= a_v.mean()
        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, 1), -1)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, -1), 1)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

    #--- Complex Envelope Input, Even Number of Points, Time Offset
    def test_c3(self):
        n = 2**6
        n_2 = n//2
        dv = 0.5
        t_ref = 100.

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j
        a_v -= a_v.mean()
        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, 1, t_ref=t_ref), -1, t_ref=t_ref)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, -1, t_ref=t_ref), 1, t_ref=t_ref)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

    #--- Complex Envelope Input, Odd Number of Points, Time Offset
    def test_c4(self):
        n = 2**6 + 1
        n_2 = n//2
        dv = 0.5
        t_ref = 100.

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j
        a_v -= a_v.mean()
        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, 1, t_ref=t_ref), -1, t_ref=t_ref)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

        a_v_ident = light.derivative_v(v_grid, light.derivative_v(v_grid, a_v, -1, t_ref=t_ref), 1, t_ref=t_ref)
        assert np.allclose(a_v_ident, a_v, equal_nan=True)

#--- Exploratory
if __name__ == "__main__":
    TestDerivativeT().test_all()
    TestDerivativeV().test_all()

    #--- Setup
    n = 2**6 + 1
    n_2 = n//2
    dt = 0.5
    dv = 1/(n*dt)
    ndv = n*dv

    #--- Plotting
    fig0 = plt.figure("Time Domain Calculus", figsize=plt.figaspect(1/2))
    fig0.clf()
    ax0 = plt.subplot2grid([1,2], [0,0])
    ax1 = plt.subplot2grid([1,2], [0,1])

    fig1 = plt.figure("Frequency Domain Calculus", figsize=plt.figaspect(1/2))
    fig1.clf()
    ax2 = plt.subplot2grid([1,2], [0,0])
    ax3 = plt.subplot2grid([1,2], [0,1])

    #--- Time Grids
    t_grid = dt*np.arange(-n_2, n - n_2)
    a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j
    a_t -= a_t.mean()

    #--- Time Domain Derivative
    da_dt = light.derivative_t(t_grid, a_t, 1)
    # a_t_ident = light.derivative_t(t_grid, da_dt, -1)

    ax0.set_title("Derivative")
    ax0.plot(t_grid, da_dt.real, label="Fourier - real")
    ax0.plot(t_grid, da_dt.imag, label="Fourier - imag")
    ax0.plot(t_grid, np.gradient(a_t, t_grid).real, '.', label="grad - real")
    ax0.plot(t_grid, np.gradient(a_t, t_grid).imag, '.', label="grad - imag")
    ax0.legend()

    #--- Time Domain Integral
    ia_it = light.derivative_t(t_grid, a_t, -1)

    ax1.set_title("Integral")
    ax1.plot(t_grid, ia_it.real, label="Fourier - real")
    ax1.plot(t_grid, ia_it.imag, label="Fourier - imag")
    ax1.plot(t_grid, (ia_it[0] + integrate.cumtrapz(a_t, t_grid, initial=0)).real, '.', label="trapz - real")
    ax1.plot(t_grid, (ia_it[0] + integrate.cumtrapz(a_t, t_grid, initial=0)).imag, '.', label="trapz - imag")
    ax1.legend()
    fig0.tight_layout()

    #--- Frequency Grids
    v_grid = dv*np.arange(-n_2, n - n_2)
    a_v = np.exp(-0.5*(v_grid/(5*dv))**2)
    if np.isreal(a_v[-1]):
        rn = 2*(n-1)
    else:
        rn = 2*(n-1) + 1
    a_v -= a_v.mean()

    #--- Frequency Domain Derivative
    da_dv = light.derivative_v(v_grid, a_v, 1)
    # a_v_ident = light.derivative_v(v_grid, da_dv, -1)

    ax2.set_title("Derivative")
    ax2.plot(v_grid, da_dv.real, label="Fourier - real")
    ax2.plot(v_grid, da_dv.imag, label="Fourier - imag")
    ax2.plot(v_grid, np.gradient(a_v, v_grid).real, '.', label="grad - real")
    ax2.plot(v_grid, np.gradient(a_v, v_grid).imag, '.', label="grad - imag")
    ax2.legend()

    #--- Frequency Domain Integral
    ia_iv = light.derivative_v(v_grid, a_v, -1)

    ax3.set_title("Integral")
    ax3.plot(v_grid, ia_iv.real, label="Fourier - real")
    ax3.plot(v_grid, ia_iv.imag, label="Fourier - imag")
    ax3.plot(v_grid, (ia_iv[0] + integrate.cumtrapz(a_v, v_grid, initial=0)).real, '.', label="trapz - real")
    ax3.plot(v_grid, (ia_iv[0] + integrate.cumtrapz(a_v, v_grid, initial=0)).imag, '.', label="trapz - imag")
    ax3.legend()

    fig1.tight_layout()


# %% Resampling

class TestResampleT():
    def test_all(self):
        if __name__ == "__main__":
            self.test_r1()
            self.test_r2()

            self.test_c1()
            self.test_c2()

    #--- Real-Valued Input, Even Number of Points
    def test_r1(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Real-Valued Input, Odd Number of Points
    def test_r2(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j*np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        n_2 = n//2
        dt = 0.5

        t_grid = dt*np.arange(-n_2, n - n_2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j*np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_t(t_grid, a_t, n_rs)
        rs_2 = light.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

class TestResampleV():
    def test_all(self):
        if __name__ == "__main__":
            self.test_r1()
            self.test_r2()

            self.test_c1()
            self.test_c2()

    #--- Real-Valued Input, Even Number of Points
    def test_r1(self):
        n = 2**6
        rn = 2*(n - 1)
        dv = 0.5

        v_grid = dv*np.arange(n)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*rn
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*rn + 1
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Real-Valued Input, Odd Number of Points
    def test_r2(self):
        n = 2**6
        rn = 2*(n - 1) + 1
        dv = 0.5

        v_grid = dv*np.arange(n)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 0j
        print(a_v[-1])
        a_v[-1] += 1j # force number of points in the time domain to be odd

        #--- Even
        n_rs = 2*rn
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*rn + 1
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        n_2 = n//2
        dv = 0.5

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j*np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        n_2 = n//2
        dv = 0.5

        v_grid = dv*np.arange(-n_2, n - n_2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j*np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = light.resample_v(v_grid, a_v, n_rs)
        rs_2 = light.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)


#--- Exploratory
if __name__ == "__main__":
    TestResampleT().test_all()
    TestResampleV().test_all()

    #--- Setup
    n = 2**6 + 1
    n_2 = n//2
    dt = 0.5
    dv = 1/(n*dt)
    ndv = n*dv

    #--- Plotting
    fig0 = plt.figure("Resampling", figsize=plt.figaspect(1/2))
    fig0.clf()
    ax0 = plt.subplot2grid([1,2], [0,0])
    ax1 = plt.subplot2grid([1,2], [0,1])

    #--- Time Grids
    t_grid = dt*np.arange(-n_2, n - n_2)
    a_t = np.exp(-0.5*(t_grid/(5*dt))**2)

    #--- Time Domain Resampling
    rs_t1 = light.resample_t(t_grid, a_t, 3*n + 1)
    rs_t2 = light.resample_t(t_grid, a_t, n - n//3)

    ax0.set_title("Time Domain")
    ax0.plot(t_grid, a_t, 'o', label="Original")
    ax0.plot(rs_t1.t_grid, rs_t1.f_t, '.', label="Resample 1")
    ax0.plot(rs_t2.t_grid, rs_t2.f_t, '.', label="Resample 2")
    ax0.legend()

    #--- Frequency Grids
    v_ref = 10
    v_grid = dv*np.arange(-n_2, n - n_2) + v_ref
    # v_grid = dv*np.arange(n)
    a_v = np.exp(-0.5*((v_grid-v_ref)/(5*dv))**2)

    #--- Frequency Domain Resampling
    rs_v1 = light.resample_v(v_grid, a_v, 3*n + 1)
    rs_v2 = light.resample_v(v_grid, a_v, n - n//3)

    ax1.set_title("Frequncy Domain")
    ax1.plot(v_grid, a_v, 'o', label="Original")
    ax1.plot(rs_v1.v_grid, rs_v1.f_v, '.', label="Resample 1")
    ax1.plot(rs_v2.v_grid, rs_v2.f_v, '.', label="Resample 2")
    ax1.legend()
    fig0.tight_layout()


# %% TFGrid

def test_TFGrid():
    test = light.TFGrid(700, 2.5, 2**6 + 1)

    #--- Complex Envelope Frequency Grid
    assert all(test.v_grid == fft.fftshift(test._v_grid))
    assert test.v_ref == test._v_ref
    assert test.dv == test._dv
    assert test.dv == test.rdv
    assert test.ndv == test.n * test.dv
    assert test.v_window == test.dv*test.n

    #--- Complex Envelope Time Grid
    assert all(test.t_grid == fft.fftshift(test._t_grid))
    assert test.t_ref == test._t_ref
    assert test.dt == test._dt
    assert test.t_window == test.n * test.dt

    #--- Real-Valued Time and Frequency Grid
    assert all(test.rv_grid == test._rv_grid)
    assert test.rv_ref == 0
    assert test.rdv == test.dv
    assert test.rndv == test.rn*test.rdv
    assert test.rv_window == len(test.rv_grid) * test.rdv
    assert all(test.rt_grid == fft.fftshift(test._rt_grid))
    assert test.rt_ref == test.t_ref
    assert test.rdt == test._rdt
    assert test.rt_window == test.t_window

    #--- rtf_grids
    #TODO: test this method

#TODO: test TFGrid.from_freq_range
#TODO: test TFGrid.from_time_freq

#--- Exploratory
if __name__ == "__main__":
    test_TFGrid()

# %% Pulse

def test_pulse_shapes():
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15

    #--- Gaussian Pulse
    test = light.Pulse.gaussian_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Sech**2 Pulse
    test = light.Pulse.sech2_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Parabolic Pulse
    test = light.Pulse.parabolic_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Lorentzian**2 Pulse
    test = light.Pulse.lorentzian2_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- CW Light
    p_p = 1
    test = light.Pulse.cw_light(v_min, v_max, n, v_0, p_p)
    assert np.isclose(p_p, test.e_p/test.t_window)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)

def test_pulse_properties():
    """
    Test Pul properties on a generated gaussian pulse.
    """
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15
    test = light.Pulse.gaussian_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)

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
    assert np.isclose(np.log(4)**0.5 / (2*pi*v_w.rms), t_w.fwhm, atol=test.dt)
    ac = test.autocorrelation()
    assert np.isclose(ac.rms, 2**0.5 * t_w.rms, atol=0)

#--- Exploratory
if __name__ == "__main__":
    test_pulse_shapes()
    test_pulse_properties()

    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 50e-15
    test = light.Pulse.gaussian_pulse(v_min, v_max, n, v_0, e_p, t_fwhm)

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

# %% Even and Odd Numbers of Points

t_grid = np.linspace(-2, 2, 10000)[:-1]
dt = np.diff(t_grid).mean()
cont_t = 0.25*np.cos(2*np.pi/2 * t_grid)
print(np.sum(cont_t**2)*dt)

n = 4
dt = 1
dv = 1/(n*dt)
test_v = np.array([1,0,0,0])
test_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(test_v*n*dv)))
# print(test_t)
print(np.sum(test_t**2)*dt)
print(np.sum(test_v**2)*dv / (np.sum(test_t**2)*dt))

n2 = 5
dv2 = dv
dt2 = 1/(n2*dv2)
test2_v = np.array([1/2,0,0,0,1/2])
test2_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(test2_v*n2*dv2)))
# print(test2_t)
print(np.sum(test2_t**2)*dt2)
print(np.sum(test2_v**2)*dv2 / (np.sum(test2_t**2)*dt2))

plt.figure("test")
plt.clf()
plt.plot(t_grid, cont_t)
plt.plot(np.arange(-(n//2), n - (n//2))*dt, test_t.real, '.', label="even", markersize=10)
plt.plot(np.arange(-(n2//2), n2 - (n2//2))*dt2, test2_t.real, '.', label="odd", markersize=10)
plt.legend()

#%% ...2

plt.figure("test")
plt.clf()
dv = 1

n = 10000
dt = 1/(n*dv)
t_grid = dt*np.arange(-(n//2), n - (n//2))
test_t = np.exp(1j*(2*np.pi/.5)*t_grid)*np.exp(1j*np.pi/5)
test_t = test_t.real
plt.plot(t_grid, test_t.real, "-", color="C0")
plt.plot(t_grid, test_t.imag, ":", color="C0")

n = 5
dt = 1/(n*dv)
t_grid = dt*np.arange(-(n//2), n - (n//2))
test1_t = np.exp(1j*(2*np.pi/.5)*t_grid)*np.exp(1j*np.pi/5)
test1_t = test1_t.real
test1_v = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(test1_t*dt)))
print(test1_v.round(3))
plt.plot(t_grid, test1_t.real, '.', color="C1")
plt.plot(t_grid, test1_t.imag, '.', color="C1")

n = 4
dt = 1/(n*dv)
t_grid = dt*np.arange(-(n//2), n - (n//2))
test2_t = np.exp(1j*(2*np.pi/.5)*t_grid)*np.exp(1j*np.pi/5)
test2_t = test2_t.real
test2_v = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(test2_t*dt)))
plt.plot(t_grid, test2_t.real, '.', color="C2")
plt.plot(t_grid, test2_t.imag, '.', color="C2")
print(test2_v.round(3))


# %% Real to Analytic

n = 1000
an = n//2 + 1

# Real
dv = 1
dt = 1/(n*dv)
t_grid = dt*np.arange(-(n//2), n - (n//2))

# test_t = np.random.normal(size=n)
test_t = np.exp(1j*2*pi*t_grid/.5).imag + 2
test_v = fft.fftshift(fft.fft(fft.ifftshift(test_t)*dt))
test_rv = fft.rfft(fft.ifftshift(test_t)*dt)

# Analytic
print(n, an)
adt = 1/(an*dv)
adt_grid = adt*np.arange(-(an//2), an - (an//2))

test_av = 2**0.5 * test_rv.copy()
test_av[0] /= 2**0.5
test_av[-1] /= 1 if n%2 else 2**0.5
test_at = fft.ifft(fft.ifftshift(test_av)*an*dv)

# Power
test1 = np.sum(test_t**2 *dt)
test2 = np.sum(np.abs(test_v)**2 * dv)
test3 = np.sum(np.abs(test_av)**2 * dv)
test4 = np.sum(np.abs(test_at)**2 * adt)

print(test2/test1)
print(test3/test1)
print(test4/test1)

plt.figure("test")
plt.clf()
plt.plot(adt*np.arange(-(an//2), an - (an//2)), test_at.real)
plt.plot(adt*np.arange(-(an//2), an - (an//2)), test_at.imag)
plt.plot(adt*np.arange(-(an//2), an - (an//2)), np.abs(test_at))
plt.plot(t_grid, test_t)

plt.figure("test2")
plt.clf()
plt.plot(t_grid, test_t**2)
plt.plot(adt*np.arange(-(an//2), an - (an//2)), np.abs(test_at)**2)

# %% Analytic to Real

an = 11
n = (an - 1)*2 #even
# n = an*2 - 1 #odd
print(n, an)

# Analytic
adt = .25
dv = 1/(an*adt)

test_av = np.random.normal(size=an) + 1j*np.random.normal(size=an)
test_av[0] = test_av[0].real
test_av[-1] = test_av[-1] if n%2 else test_av[-1].real
test_at = fft.ifft(fft.ifftshift(test_av*an*dv))

# Real
dt = 1/(n*dv)

test_rv = 2**-0.5 * test_av.copy()
test_rv[0] *= 2**0.5
test_rv[-1] *= 1 if n%2 else 2**0.5

test_t = fft.irfft(test_rv*n*dv, n=n)
test_v = fft.fft(test_t*dt)

# Power
test1 = np.sum(test_t**2 *dt)
test2 = np.sum(np.abs(test_v)**2 * dv)
test3 = np.sum(np.abs(test_av)**2 * dv)
test4 = np.sum(np.abs(test_at)**2 * adt)

print(test2/test1)
print(test3/test1)
print(test4/test1)


# %%
# # # test_grid = TFGrid(100e12, 1e12, 50)
# # # test = Pulse.from_TFGrid(test_grid)

# # # test1 = Pulse(100e12, 1e12, 50)

# # %% Testing
# import matplotlib.pyplot as plt

# test = Pulse.gaussian_pulse(150e12, 500e12, 2**6, 325e12, 1, 30e-15)
# # test = Pulse.cw_light(150e12, 500e12, 2**12, 200.0e12, 1)

# print(np.sum(test.p_v * test.dv))
# print(np.sum(test.p_t * test.dt))
# print(np.sum(test.ra_t**2 * test.rdt))


# # %%

# test1 = resample_t(test.rt_grid, test.ra_t, 2**15)
# print(np.sum(test1.f_t**2 * test1.dt))

# test4 = fft.irfft(fft.rfft(test1.f_t) * np.exp(1j*2*pi*(50e-15)*(test.dv*np.arange(test1.t_grid.size//2 + 1))))

# plt.figure("test0")
# plt.clf()
# plt.plot(1e12*test1.t_grid, test4)
# plt.plot(1e12*test1.t_grid, test1.f_t)
# plt.plot(1e12*test.rt_grid, test.ra_t, '.')
# plt.plot(1e12*test.t_grid, np.abs(test.a_t))
# # plt.semilogy(1e12*test.t_grid, test.p_t)
# # plt.semilogy(1e12*test.rt_grid, test.ra_t**2)
# # plt.plot(1e12*test.t_grid, 1e-12*test.vg_t)

# plt.figure("test")
# plt.clf()
# plt.plot(1e12*test.rt_grid, test.rp_t)
# plt.plot(1e12*test.t_grid, test.p_t)
# # plt.semilogy(1e12*test.t_grid, test.p_t)
# # plt.semilogy(1e12*test.rt_grid, test.ra_t**2)
# # plt.plot(1e12*test.t_grid, 1e-12*test.vg_t)

# test2 = resample_v(test.v_grid, test.a_v, 2**12)
# print(np.sum(np.abs(test2.f_v)**2 * test2.dv))

# test3 = resample_v(test.rv_grid, 2**0.5*fft.rfft(test.ra_t*test.rdt), 2*(2**12 + -1), rn_0=test.rn)
# print(np.sum(np.abs(test3.f_v)**2 * test3.dv))

# plt.figure("test2")
# plt.clf()
# # plt.plot(1e-12*test.v_grid, test.p_v)
# plt.semilogy(1e-12*test3.v_grid, np.abs(test3.f_v)**2, '.-')
# plt.semilogy(1e-12*test2.v_grid, np.abs(test2.f_v)**2, '.-')
# plt.semilogy(1e-12*test.v_grid, test.p_v, '.')
# # plt.plot(1e-12*test.v_grid, 1e12*test.tg_v, '.')

# # # plt.figure("test3")
# # # plt.clf()
# # # plt.semilogy(1e-12*test.v_grid, test.p_v)
# # # # plt.plot(1e-12*test.v_grid, 1e12*test.tg_v, '.')


# # # # #%%
# # # # # test.phi_v += 2*pi * 1000*1e-15**2 * 1/2*(test.v_grid - test.v_0)**2
# # # # # test_spec = test.spectrogram()
# # # # # test_spec = test.spectrogram(t_fwhm=50e-15)
# # # # test_spec = test.spectrogram(n_t="equal")

# # # # plt.figure("test3")
# # # # plt.clf()
# # # # plt.imshow(10*np.log10(test_spec.spg), aspect="auto", origin="lower", extent=test_spec.extent, cmap=plt.cm.nipy_spectral)
# # # # # plt.imshow(10*np.log10(test_spec.spg), aspect="auto", origin="lower", extent=(-100, 100, -100, 100), cmap=plt.cm.nipy_spectral)

# # # # %%
# # # ac1 = test.autocorrelation()
# # # ac2 = test.autocorrelation(n=2**15)

# # # plt.figure("test4")
# # # plt.clf()
# # # plt.plot(ac2.t_grid, ac2.ac_t)
# # # plt.plot(ac1.t_grid, ac1.ac_t, '.')

# # %%
# test = TFGrid(10, .01, 10)
