# -*- coding: utf-8 -*-
"""
Models for simulating the propagation of optical pulses.

Notes
-----
Multiplications of functions in the time domain (operations intrinsic to
nonlinear optics) are equivalent to convolutions in the frequency domain and
vice versa. The support of a convolution is the sum of the support of its
parts. Thus, 2nd and 3rd order processes in the time domain need 2x and 3x
number of points in the frequency domain to avoid aliasing.

`Pulse` objects only initialize the minimum number of points necessary to
represent the real-valued time domain pulse (i.e., 1x). While this minimizes
the numerical complexity of individual nonlinear operations, aliasing
introduces systematic error and can even increase the total simulation time by
forcing shorter step sizes. More points can be generated for a specific
`Pulse` object by setting an `alias` parameter greater than 1 during its
initialization or through its :py:meth:`~pynlo.utility.TFGrid.rtf_grids`
method. Anti-aliasing is not always necessary as phase matching can suppress
the aliased interactions, but it is best practice to verify that behavior on a
case-by-case basis.

"""

__all__ = ["SM_UPE"]


# %% Imports

import collections

import numpy as np
from scipy.constants import c, pi
from numba import njit

from pynlo.light import Pulse
from pynlo.media import Mode
from pynlo.utility import fft


# %% Collections

SimulationResult = collections.namedtuple("SimulationResult", ["pulse", "z", "a_t", "a_v"])


# %% Routines

@njit(parallel=True)
def exp(x):
    """JIT compiled exponential function."""
    return np.exp(x)

@njit
def fdd(f, dx, idx):
    """JIT compiled 2nd order finite difference derivative."""
    #---- Right Bound
    if idx==f.size-1:
        # Derivative on the right-most edge
        return (3*f[idx] - 4*f[idx-1] + f[idx-2])/(2*dx)

    #---- Left Bound
    if idx==0:
        # Derivative on the left-most edge
        return (-3*f[idx] + 4*f[idx+1] - f[idx+2])/(2*dx)

    #---- Central
    return (f[idx+1] - f[idx-1])/(2*dx)

@njit(parallel=True)
def l2_error(a_RK4, a_RK3):
    """JIT compiled l2 error norm."""
    l2_norm = np.sum(np.real(a_RK4)**2 + np.imag(a_RK4)**2)**0.5
    rel_error = (a_RK4 - a_RK3)/l2_norm
    return np.sum(np.real(rel_error)**2 + np.imag(rel_error)**2)**0.5


# %% Classes

class SM_UPE():
    """
    A model for simulating single-mode pulse propagation.

    This model can simultaneously simulate both 2nd and 3rd order nonlinear
    effects.

    Parameters
    ----------
    pulse : pynlo.light.Pulse
        A `Pulse` object containing the properties of the input pulse and of
        the underlying time and frequency grids.
    mode : pynlo.media.Mode
        A `Mode` object containing the linear and nonlinear properties of
        the waveguide mode.

    """

    def __init__(self, pulse, mode):
        """
        Initialize a single-mode model for simulating pulse propagation.

        Parameters
        ----------
        pulse : pynlo.light.Pulse
            A `Pulse` object containing the properties of the input pulse and
            of the underlying time and frequency grids.
        mode : pynlo.media.Mode
            A `Mode` object containing the linear and nonlinear properties of
            the waveguide.
        """
        assert isinstance(pulse, Pulse)
        assert isinstance(mode, Mode)
        assert (pulse.v_grid==mode.v_grid).all(), ("The pulse and mode must be defined"
                                                   " over the same frequency grid")
        if mode.rv_grid is not None:
            assert (pulse.rv_grid==mode.rv_grid).all(), ("The pulse and mode must be defined"
                                                         " over the same frequency grid")
        self.pulse_in = pulse
        self.mode = mode

        #---- Analytic Grids
        # Frequency Grids
        self.n_points = self.pulse_in.n
        self.v_grid = self.pulse_in.v_grid
        self.w_grid = 2*pi*self.v_grid
        self.dw = 2*pi*self.pulse_in.dv

        # Time Grid
        self.t_grid = self.pulse_in.t_grid
        self.dt = self.pulse_in.dt

        # Wavelength Grid
        self.l_grid = c/self.v_grid
        self.dv_dl = self.v_grid**2/c # power density conversion factor

        #---- Carrier Resolved Grids
        # CR Slice
        self.cr_slice = self.pulse_in.rn_slice
        self.cr_pre_slice = slice(None, self.pulse_in.rn_range.min())
        self.cr_post_slice = slice(self.pulse_in.rn_range.max() + 1, None)

        # Time Grid
        self.nl_n_points = self.pulse_in.rn
        self.nl_dt = self.pulse_in.rdt

        #---- Waveguide Parameters
        # Gain
        self.alpha = self.mode.alpha()

        # Wavenumber
        self.beta = self.mode.beta()
        self.v0_idx = self.pulse_in.v0_idx
        beta1_v0 = fdd(self.beta, self.dw, self.v0_idx)
        self.beta_cm = self.beta - beta1_v0*self.w_grid # comoving frame

        # Propagation Constant (comoving frame)
        if self.alpha is not None:
            self.kappa_cm = (self.beta_cm - self.alpha**2/(8*self.beta)) + 0.5j*self.alpha
        else:
            self.kappa_cm = self.beta_cm

        # Nonlinear Parameters
        self.g2 = self.mode.g2()
        self.g3 = self.mode.g3()
        self.r3 = self.mode.r3()

        #---- Implementation Details
        self._update_alpha = self._update_beta = True
        self.update_linearity() # initialize linearity
        self._update_alpha, self._update_beta = self.mode.z_dep_linearity
        self._z_linear = self._update_alpha or self._update_beta

        self._update_g2 = self._update_g2_inv = self._update_g3 = self._update_r3 = True
        self.update_nonlinearity() # initialize nonlinearity
        (self._update_g2, self._update_g2_inv,
         self._update_g3, self._update_r3) = self.mode.z_dep_nonlinearity
        self._z_nonlinear = (self._update_g2 or self._update_g2_inv
                             or self._update_g3 or self._update_r3)

        self._1j_w_grid = 1j * self.w_grid
        self._nl_v = np.zeros_like(self.v_grid, dtype=complex)
        self._nl_a_v = np.zeros_like(self.pulse_in.rv_grid, dtype=complex)

    def estimate_step_size(self, a_v=None, z=None, local_error=1e-6, dz=10e-6, n=1, db=False):
        """
        Estimate a more optimal step size, relative to the local error, given
        a test step size `dz`.

        Call this method with a larger `n` to iteratively approach the
        optimum step size for a given `local_error`.

        Parameters
        ----------
        a_v : array_like of complex
            The spectral amplitude of the pulse.
        z : float
            The position along the waveguide.
        local_error : float, optional
            The relative local error. The default is 1e-6.
        dz : float, optional
            The step size. The default is 10e-6.
        n : int, optional
            The number of times the algorithm is iteratively executed
        db : bool, optional
            Flag which turns on printing of intermediate results.

        Returns
        -------
        dz : float
            The new step size.

        """
        if a_v is None:
            a_v = self.pulse_in.a_v
        if z is None:
            z = self.mode.z

        for _ in range(n):
            #---- Integrate by dz
            a_RK4_v, a_RK3_v, _ = self.integrate(a_v, z, dz)

            #---- Estimate the Relative Local Error
            est_error = l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error/local_error)**0.25
            if db: print("dz={:.3g},\t error={:.3g}".format(dz, est_error))

            #---- Update Step Size
            dz = dz/min(2, max(error_ratio, 0.5))
        return dz

    def simulate(self, z_grid, dz=None, local_error=1e-6, n_records=None, plot=None):
        """
        Simulate the propagation of the pulse along the waveguide.

        Parameters
        ----------
        z_grid : array_like of floats
            The positions along the waveguide at which to solve for the pulse
            spectrum. An adaptive step size algorithm is used to propagate
            between these points. If only the end point is given the starting
            point is assumed to be at the origin.
        dz : float, optional
            The initial step size. If ``None``, one will be estimated.
        local_error : float, optional
            The target relative local error for the adaptive step size
            algorithm. The default is 1e-6.
        n_records : None or int, optional
            The number of points to return. If set, the positions will be
            linearly spaced between the first and last points of `z_grid`. The
            default is to return all points as defined in `z_grid`, including
            the starting position.
        plot : None or string, optional
            A flag that selects the type of plot for real-time visualization
            of the simulation. The options are "frq", "time", or "wvl" and
            correspond to the frequency, time, and wavelength domains. The
            default is to run the simulation without real-time plotting.

        Returns
        -------
        pulse : pynlo.light.Pulse
            The output pulse. This object can be used as the input to another
            waveguide.
        z : ndarray of float
            The points along the waveguide at which the pulse spectrum (`a_v`)
            and envelope (`a_t`) are returned.
        a_t : ndarray of complex
            The root-power complex envelope of the pulse along the waveguide.
        a_v : ndarray of complex
            The root-power spectrum of the pulse along the waveguide.
        """
        #---- Z Sample Space
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.size==1:
            # Only the end point was given, assuming that the start point is the origin
            z_grid = np.append(0.0, z_grid)
        z = z_grid[0]

        if n_records is None:
            n_records = z_grid.size
            z_record = z_grid
        else:
            assert n_records>=2, "The output must include atleast 2 points."
            z_record = np.linspace(z_grid.min(), z_grid.max(), n_records)
            z_grid = np.unique(np.append(z_grid, z_record))

        #---- Setup Pulse
        pulse_out = Pulse.FromTFGrid(self.pulse_in, a_v=self.pulse_in.a_v)
        # Frequency Domain
        a_v_record = np.empty((n_records,pulse_out.n), dtype=complex)
        a_v_record[0,:] = pulse_out.a_v
        # Time Domain
        a_t_record = np.empty((n_records,pulse_out.n), dtype=complex)
        a_t_record[0,:] = pulse_out.a_t

        #---- Plotting
        if plot is not None:
            assert (plot in ["frq", "time", "wvl"]), ("Plot choice '{:}' is"
                                                      " unrecognized").format(plot)
            # Setup Plots
            self._setup_plots(plot, pulse_out, z)

        #---- Step Size
        if dz is None:
            dz = self.estimate_step_size(pulse_out.a_v, z, local_error, n=10)
            print("Initial Step Size:\t{:.3g}".format(dz))

        #---- Propagate
        k5_v = None
        for z_stop in z_grid[1:]:
            #---- Step
            (pulse_out.a_v, z, dz, k5_v) = self.propagate(
                pulse_out.a_v,
                z, z_stop, dz,
                local_error,
                k5_v=k5_v)

            #---- Record
            if z in z_record:
                idx = np.flatnonzero(z==z_record)
                a_t_record[idx,:] = pulse_out.a_t
                a_v_record[idx,:] = pulse_out.a_v

                #---- Plot
                if plot is not None:
                    # Update Plots
                    self._update_plots(plot, pulse_out, z)

                    if z==z_grid[-1]:
                        # End animation with the last step
                        for artist in self._artists:
                            artist.set_animated(False)
            sim_res = SimulationResult(
                pulse=pulse_out, z=z_record, a_t=a_t_record, a_v=a_v_record)
        return sim_res

    def propagate(self, a_v, z_start, z_stop, dz, local_error, k5_v=None):
        """
        Propagates the given pulse spectrum from `z_start` to `z_stop` using
        an adaptive step size algorithm.

        The algorithm is based on an embedded 4th order Runge-Kutta method in
        the interaction picture (ERK4(3)-IP) [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z_start : float
            The starting position.
        z_stop : float
            The stopping position.
        dz : float
            The initial step size.
        local_error : float
            The target relative local error for the adaptive step size
            algorithm.
        k5_v : ndarray of complex, optional
            A parameter calculated during the preceding integration step. When
            included, it allows calculation with one less call to the
            nonlinear operator. The default is None.

        Returns
        -------
        a_v : ndarray of complex
            The final root-power spectrum of the pulse.
        z : float
            The final position along the waveguide.
        dz : float
            The final step size.
        k5_v : ndarray of complex
            A parameter calculated during the final integration step.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        z = z_start
        while z < z_stop:
            z_next = z + dz
            dz_next = dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz = z_next - z
            else:
                final_step = False

            #---- Integrate by dz
            a_RK4_v, a_RK3_v, k5_v_next = self.integrate(a_v, z, dz, k5_v=k5_v)

            #---- Estimate Relative Local Error
            est_error = l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error/local_error)**0.25

            #---- Propagate Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                dz = dz/2
            else:
                # Update parameters for the next loop
                a_v = a_RK4_v
                z = z_next
                if (not final_step) or (error_ratio > 1):
                    dz_next = dz / max(error_ratio, 0.5)
                dz = dz_next
                k5_v = k5_v_next
        return a_v, z, dz, k5_v

    def integrate(self, a_v, z, dz, k5_v=None):
        """
        Integrates the given pulse spectrum over a step size of `dz`.

        This uses an embedded 4th order Runge-Kutta method in the interaction
        picture (ERK4(3)-IP) [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The starting position.
        dz : float
            The step size.
        k5_v : ndarray of complex, optional
            The nonlinear response of the solution from the preceding
            integration step. When included, it allows calculation of the
            integral with one less call to the nonlinear operator.

        Returns
        -------
        a_RK4_v : ndarray of complex
            The 4th order result.
        a_RK3_v : ndarray of complex
            The 3rd order result.
        k5_v : ndarray of complex
            The nonlinear response of the 4th order result.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        #---- k1
        if self._z_linear and self.mode.z != z: # check if previous step was rejected
            self.update_linearity(z)

        if k5_v is None:
            if self._z_nonlinear: self.update_nonlinearity(z)
            k5_v = self._nonlinear_operator(a_v)
        IP_in_v = self.linear_operator(0.5*dz)
        aI_v = IP_in_v * a_v # into interaction picture
        kI1_v = IP_in_v * k5_v # into interaction picture

        #---- k2
        if self._z_nonlinear: self.update_nonlinearity(z+0.5*dz)
        aI2_v = aI_v + 0.5*dz * kI1_v
        kI2_v = self._nonlinear_operator(aI2_v)

        #---- k3
        aI3_v = aI_v + 0.5*dz * kI2_v
        kI3_v = self._nonlinear_operator(aI3_v)

        #---- k4
        if self._z_linear:
            self.update_linearity(z+dz)
            IP_out_v = self.linear_operator(0.5*dz)
        else:
            IP_out_v = IP_in_v
        if self._z_nonlinear:
            self.update_nonlinearity(z+dz)

        aI4_v = aI_v + dz*kI3_v
        a4_v = IP_out_v * aI4_v # out of interaction picture
        k4_v = self._nonlinear_operator(a4_v)

        #---- RK4
        bI_v = aI_v + dz/6.0 * (kI1_v + 2.0*(kI2_v + kI3_v))
        b_v = IP_out_v * bI_v # out of interaction picture
        a_RK4_v = b_v + dz/6.0 * k4_v

        #---- k5
        k5_v = self._nonlinear_operator(a_RK4_v)

        #---- RK3
        a_RK3_v = b_v + dz/30.0 * (2.0*k4_v + 3.0*k5_v)
        return a_RK4_v, a_RK3_v, k5_v

    #---- Operators
    def linear_operator(self, dz):
        """
        The frequency-domain linear operator integrated over the given step
        size.

        Parameters
        ----------
        dz : float
            The step size.

        Returns
        -------
        ndarray of complex

        """
        # Linear Operator
        l_v = -1j*dz * self.kappa_cm
        return exp(l_v)

    def nonlinear_operator(self, a_v):
        """
        The frequency-domain nonlinear response of the given pulse spectrum.

        This is only the spectral change per unit propagation length. Estimate
        the actual change by integrating this value over some step size.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        """
        #---- Setup
        self._nl_v = np.zeros(self.n_points, dtype=complex)
        self._nl_a_v[self.cr_slice] = a_v
        nl_a_t = fft.irfft(self._nl_a_v, fsc=self.nl_dt * 2**0.5, n=self.nl_n_points)
        nl_a2_t = nl_a_t * nl_a_t

        #---- 2nd Order Nonlinearity
        if self.g2 is not None:
            nl_a2_v = fft.rfft(nl_a2_t, fsc=self.nl_dt * 2**0.5)
            if self.g2_inv == 1:
                self._nl_v -= self.g2 * nl_a2_v[self.cr_slice]
            else:
                self._nl_v += self.g2 * nl_a2_v[self.cr_slice]

        #---- 3rd Order Nonlinearity
        if self.g3 is not None:
            if self.r3 is not None:
                nl_a2_v = (fft.rfft(nl_a2_t, fsc=self.nl_dt) if self.g2 is None
                           else nl_a2_v * 2**-0.5)
                nl_a2r3_v = nl_a2_v * self.r3
                nl_a2_t = fft.irfft(nl_a2r3_v, fsc=self.nl_dt, n=self.nl_n_points)
            nl_a3_t = nl_a_t * nl_a2_t
            nl_a3_v = fft.rfft(nl_a3_t, fsc=self.nl_dt * 2**0.5)
            self._nl_v -= self.g3 * nl_a3_v[self.cr_slice]

        #---- Nonlinear Response
        return self._1j_w_grid * self._nl_v # minus sign included in _nl_v

    def nonlinear_operator_svd(self, a_v):
        """
        The frequency-domain nonlinear response of the given pulse spectrum,
        calculated using the nonlinear parameters given in separable form.

        This is only the spectral change per unit propagation length. Estimate
        the actual change by integrating this value over some step size.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        Notes
        -----
        TODO: describe expected separable form

        """
        #---- Setup
        self._nl_v = np.zeros(self.n_points, dtype=complex)

        #---- 2nd Order Nonlinearity
        if self.g2 is not None:
            nl_a2_t = np.zeros(self.nl_n_points, dtype=float)
            for g2_internal in self.g2[1:]:
                self._nl_a_v[self.cr_slice] = a_v * g2_internal
                nl_a_t = fft.irfft(self._nl_a_v, fsc=self.nl_dt * 2**0.5, n=self.nl_n_points)
                nl_a2_t += nl_a_t * nl_a_t
            nl_a2_v = fft.rfft(nl_a2_t, fsc=self.nl_dt * 2**0.5)
            if self.g2_inv==1:
                self._nl_v -= self.g2[0] * nl_a2_v[self.cr_slice]
            else:
                self._nl_v += self.g2[0] * nl_a2_v[self.cr_slice]

        #---- 3rd Order Nonlinearity
        if self.g3 is not None:
            nl_a3_t = np.zeros(self.nl_n_points, dtype=float)
            for g3_internal in self.g3[1:]:
                self._nl_a_v[self.cr_slice] = a_v * g3_internal
                nl_a_t = fft.irfft(self._nl_a_v, fsc=self.nl_dt * 2**0.5, n=self.nl_n_points)
                nl_a2_t = nl_a_t * nl_a_t
                if self.r3 is not None:
                    nl_a2_v = fft.rfft(nl_a2_t, fsc=self.nl_dt)
                    nl_a2r3_v = nl_a2_v * self.r3
                    nl_a2_t = fft.irfft(nl_a2r3_v, fsc=self.nl_dt, n=self.nl_n_points)
                nl_a3_t += nl_a_t * nl_a2_t
            nl_a3_v = fft.rfft(nl_a3_t, fsc=self.nl_dt * 2**0.5)
            self._nl_v -= self.g3[0] * nl_a3_v[self.cr_slice]

        #---- Nonlinear Response
        return self._1j_w_grid * self._nl_v # minus sign included in _nl_v

    #---- Z-Dependency
    def update_linearity(self, z=None):
        """
        Update all z-dependent linear parameters.

        Parameters
        ----------
        z : float
            The position along the waveguide.

        """
        if z is not None:
            self.mode.z = z

        #---- Gain
        if self._update_alpha:
            self.alpha = self.mode.alpha()

        #---- Phase
        if self._update_beta:
            self.beta = self.mode.beta()
            beta1_v0 = fdd(self.beta, self.dw, self.v0_idx)
            # Beta in comoving frame
            self.beta_cm = self.beta - beta1_v0*self.w_grid

        #---- Propagation Constant
        if self.alpha is not None:
            self.kappa_cm = (self.beta_cm - self.alpha**2/(8*self.beta)) + 0.5j*self.alpha
        else:
            self.kappa_cm = self.beta_cm

    def update_nonlinearity(self, z=None):
        """
        Update all z-dependent nonlinear parameters.

        Parameters
        ----------
        z : float
            The position along the waveguide.

        """
        if z is not None:
            self.mode.z = z

        #---- 2nd Order
        if self._update_g2:
            self.g2 = self.mode.g2()
            self._g2_dim = len(self.g2.shape) if self.g2 is not None else 0
        if self._update_g2_inv:
            self.g2_inv = self.mode.g2_inv()

        #---- 3rd Order
        if self._update_g3:
            self.g3 = self.mode.g3()
            self._g3_dim = len(self.g3.shape) if self.g3 is not None else 0
        if self._update_r3:
            self.r3 = self.mode.r3()

        #---- Select Nonlinear Operator
        if self._update_g2 or self._update_g3:
            if (self._g2_dim >=2) and (self._g3_dim >= 2):
                # Separable 2nd and 3rd order terms
                self._nonlinear_operator = self.nonlinear_operator_svd
            elif (self._g2_dim >=2):
                # Separable 2nd order terms
                self._nonlinear_operator = self.nonlinear_operator_svd
                if self.g3 is not None:
                    self.g3 = [self.g3, complex(1.0)]
                    self._g3_dim = 2
            elif (self._g3_dim >= 2):
                # Separable 3rd order terms
                self._nonlinear_operator = self.nonlinear_operator_svd
                if self.g2 is not None:
                    self.g2 = [self.g2, complex(1.0)]
                    self._g2_dim = 2
            else:
                # Standard nonlinearity terms
                self._nonlinear_operator = self.nonlinear_operator


    #---- Plotting
    def _setup_plots(self, plot, pulse_out, z):
        """
        Initialize a figure for real-time visualization of a simulation.

        Parameters
        ----------
        plot : string
            The type of plot. The options are "frq", "time", or "wvl" and
            correspond to the frequency, time, and wavelength domains.
        pulse_out : pynlo.light.Pulse
            The pulse to be plotted.
        z : float
            The position along the waveguide.

        """
        # Import if needed
        try:
            self._plt
        except AttributeError:
            import matplotlib.pyplot as plt
            self._plt = plt

        #---- Figure and Axes
        self._rt_fig = self._plt.figure("Real-Time Simulation", clear=True)
        self._ax_0 = self._plt.subplot2grid((2,1), (0,0), fig=self._rt_fig)
        self._ax_1 = self._plt.subplot2grid((2,1), (1,0), sharex=self._ax_0, fig=self._rt_fig)
        self._rt_fig.show()

        #---- Time Domain
        if plot=="time":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e12*self.t_grid,
                pulse_out.p_t,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e12*self.t_grid,
                1e-12*pulse_out.vg_t,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Instantaneous Power")
            self._ax_0.set_ylabel("J / s")
            self._ax_0.set_xlabel("Delay (ps)")
            self._ax_1.set_ylabel("Frequency (THz)")
            self._ax_1.set_xlabel("Delay (ps)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.v_grid.max()-self.v_grid.min())
            self._ax_1.set_ylim(
                top=1e-12*(self.v_grid.max() + excess),
                bottom=1e-12*(self.v_grid.min() - excess))

        #---- Frequency Domain
        if plot=="frq":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e-12*self.v_grid,
                pulse_out.p_v,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e-12*self.v_grid,
                1e12*pulse_out.tg_v,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Power Spectrum")
            self._ax_0.set_ylabel("J / Hz")
            self._ax_0.set_xlabel("Frequency (THz)")
            self._ax_1.set_ylabel("Delay (ps)")
            self._ax_1.set_xlabel("Frequency (THz)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.t_grid.max()-self.t_grid.min())
            self._ax_1.set_ylim(
                top=1e12*(self.t_grid.max() + excess),
                bottom=1e12*(self.t_grid.min() - excess))

        #---- Wavelength Domain
        if plot=="wvl":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e9*self.l_grid,
                self.dv_dl * pulse_out.p_v,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e9*self.l_grid,
                1e12*pulse_out.tg_v,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Power Spectrum")
            self._ax_0.set_ylabel("J / m")
            self._ax_0.set_xlabel("Wavelength (nm)")
            self._ax_1.set_ylabel("Delay (ps)")
            self._ax_1.set_xlabel("Wavelength (nm)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.t_grid.max()-self.t_grid.min())
            self._ax_1.set_ylim(
                top=1e12*(self.t_grid.max() + excess),
                bottom=1e12*(self.t_grid.min() - excess))

        #---- Z Label
        #TODO: change to plt.barh, progress bar
        self._z_label = self._ax_1.legend(
            [],[],
            title='z = {:.6g} m'.format(z),
            loc=9,
            labelspacing=0,
            framealpha=1,
            shadow=False)
        self._z_label.set_animated(True)

        #---- Layout
        self._rt_fig.tight_layout()
        self._rt_fig.canvas.draw()

        #---- Blit
        self._artists = (self._ln_pwr, self._ln_phs, self._z_label)

        self._rt_fig_bkg_0 = self._rt_fig.canvas.copy_from_bbox(self._ax_0.bbox)
        self._rt_fig_bkg_1 = self._rt_fig.canvas.copy_from_bbox(self._ax_1.bbox)

    def _update_plots(self, plot, pulse_out, z):
        """
        Update the figure used for real-time visualization of a simulation.

        Parameters
        ----------
        plot : string
            The type of plot. The options are "frq", "time", or "wvl" and
            correspond to the frequency, time, and wavelength domains.
        pulse_out : pynlo.light.Pulse
            The pulse to be plotted.
        z : float
            The position along the waveguide.

        """
        #---- Restore Background
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_0)
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_1)

        #---- Update Data
        if plot=="time":
            self._ln_pwr.set_data(
                1e12*self.t_grid,
                pulse_out.p_t)
            self._ln_phs.set_data(
                1e12*self.t_grid,
                1e-12*pulse_out.vg_t)

        if plot=="frq":
            self._ln_pwr.set_data(
                1e-12*self.v_grid,
                pulse_out.p_v)
            self._ln_phs.set_data(
                1e-12*self.v_grid,
                1e12*pulse_out.tg_v)

        if plot=="wvl":
            self._ln_pwr.set_data(
                1e9*self.l_grid,
                self.dv_dl * pulse_out.p_v)
            self._ln_phs.set_data(
                1e9*self.l_grid,
                1e12*pulse_out.tg_v)

        #---- Update Z Label
        self._z_label.set_title('z = {:.6g} m'.format(z))

        #---- Blit
        for artist in self._artists:
            artist.axes.draw_artist(artist)

        self._rt_fig.canvas.blit(self._ax_0.bbox)
        self._rt_fig.canvas.blit(self._ax_1.bbox)
        self._rt_fig.canvas.flush_events()
