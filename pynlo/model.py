# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = ["UPE"]

import numpy as np
from scipy import constants
from numba import njit

from pynlo.light import Pulse
from pynlo.media import Mode
from pynlo.utility import fft


# %% Constants

c = constants.c
pi = constants.pi


# %% Routines

@njit(parallel=True)
def exp(x):
    """ Accelerated exponentiation """
    return np.exp(x)

@njit
def fdd(y, dx, idx):
    """ Finite difference derivative """
    return (y[idx+1] - y[idx-1])/(2*dx)


# %% Classes

class UPE():
    """

    Notes
    -----
    Aliasisng...
    Determines whether the number of points used to calculate the
    nonlinear terms are expanded in order to avoid aliasing. For the
    second order nonlinearity 2x number of points is required, and 3x
    number of points is required for the third order nonlinearity.
    Turning off antialiasing allows for a faster calculation, but the
    aliased components could introduce systematic error. Antialiasing
    is not always necessary as phase matching suppresses interactions
    with large phase mismatch, however this must be verified on a case
    by case basis. The default is False.
    """
    def __init__(self, pulse, mode):
        """
        ...single mode unidirectional propagation equation...
        both 3rd and 2nd order nonlinearities...

        Parameters
        ----------
        pulse : pynlo.light.Pulse
            DESCRIPTION.
        mode : pynlo.media.Mode
            DESCRIPTION.
        """
        assert isinstance(pulse, Pulse)
        assert isinstance(mode, Mode)
        assert (pulse.v_grid == mode.v_grid), ("The pulse and mode must be"
                                               " defined over the same frequency grid")
        if mode.rv_grid is not None:
            assert (pulse.rv_grid == mode.rv_grid), ("The pulse and mode must be"
                                                     " defined over the same frequency grid")
        self.pulse_in = pulse
        self.mode = mode

        #---- Analytic Grids
        # Frequency Grids
        self.v_grid = self.pulse_in.v_grid
        self.w_grid = 2*pi*self.v_grid
        self.dw = 2*pi*self.pulse_in.dv

        # Time Grid
        self.t_grid = self.pulse_in.t_grid
        self.dt = self.pulse_in.dt

        # Wavelength Grid
        self.l_grid = c/self.v_grid
        self.dv_dl = self.v_grid**2/c # density conversion factor

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
        self.beta1_v0 = fdd(self.beta, self.dw, self.v0_idx)
        self.beta_cm = self.beta - self.beta1_v0*self.w_grid # comoving frame

        # Propagation Constant (comoving frame)
        if self.alpha is not None:
            self.kappa_cm = self.beta_cm + 0.5j*self.alpha
        else:
            self.kappa_cm = self.beta_cm

        # Nonlinear Parameters
        self.g2 = self.mode.g2()
        self.g3 = self.mode.g3()
        self.r3 = self.mode.r3()

        #---- Implementation Details
        self._update_alpha = callable(self.mode._alpha)
        self._update_beta = callable(self.mode._beta)
        self._update_kappa = self._update_alpha or self._update_beta
        self._update_g2 = callable(self.mode._g2)
        self._update_g3 = callable(self.mode._g3)
        self._update_r3 = callable(self.mode._r3)

        self._1j_w_grid = 1j * self.w_grid
        self._nl_v = np.zeros_like(self.v_grid, dtype=complex)
        self._nl_a_v = np.zeros_like(self.pulse_in.nl_v_grid, dtype=complex)

    def estimate_step_size(self, a_v, z, local_error=1e-6, dz=1e-5):
        '''Estimate the optimal step size after integrating by a test dz'''
        #--- Integrate by dz
        a_RK4_v, a_RK3_v, _ = self.integrate(
            a_v,
            z,
            dz)

        #--- Estimate the Relative Local Error
        L2_norm = np.sum(a_RK4_v.real**2 + a_RK4_v.imag**2)**0.5
        rel_error = (a_RK4_v - a_RK3_v)/L2_norm
        L2_error = np.sum(rel_error.real**2 + rel_error.imag**2)**0.5
        error_ratio = (L2_error/local_error)**0.25
        #print('{:.3g}'.format(error_ratio))

        dz = dz * 0.5*(1 + 1/min(2, max(error_ratio, 0.5))) # approach the optimum step size
        return dz

    def simulate(self, z_grid, dz=None, local_error=1e-6, plot=None, record_interval=1):
        """
        Parameters
        ----------
        z_grid : array of floats
            The positions along the waveguide at which to solve for the pulse.
            An adaptive step size algorithm is used to propagate between these
            points. The `record_interval` parameter determines the subset of
            these points that are returned.
        dz : float, optional
            The initial step size. If ``None``, one will be estimated.
        local_error : float, optional
            The relative local error of the adaptive step size algorithm. The
            default is 1e-6.
        plot : None or str, optional
            Plots the results of the simulation in real time. Only one of the
            three domains may be selected during a run.
        record_interval : int, optional
            Determines the interval at which steps taken in `z_grid` are
            returned. Use this to only return the pulse at every
            `record_interval` point in `z_grid`. The default is 1.

        Returns
        -------
        t_grid : array of float
            The time grid of the output pulses.
        a_t_record : array of complex
            The root-power complex envelope of the output pulses.
        v_grid : array of float
            The frequency grid of the output pulses.
        a_v_record
            The root-power spectrum of the output pulses.
        z_record
            The points along the waveguide at which the pulses were returned.
        """
        #---- Z Sample Space
        z_grid = np.asarray(z_grid, dtype=float)
        z = z_grid[0]
        n_records = (z_grid.size-2)//record_interval + 2
        z_record = np.empty((n_records,), dtype=complex)
        z_record[0] = z

        #---- Optical Pulse
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
            # Import if needed
            try:
                self._plt
            except AttributeError:
                import matplotlib.pyplot as plt
                self._plt = plt
            # Setup Plots
            self.setup_plots(plot, pulse_out, z)

        #---- Step Size
        if dz is None:
            dz = self.estimate_step_size(pulse_out.a_v, z, local_error)
            print("Initial Step Size:\t{:.3g}".format(dz))

        #---- Propagate
        k5_v = None
        steps = z_grid.size
        for idx in range(1, steps):
            #---- Step
            z_stop = z_grid[idx]
            (pulse_out.a_v, z, dz, k5_v) = self.propagate(
                pulse_out.a_v,
                z,
                z_stop,
                dz,
                local_error,
                k5_v=k5_v)

            #---- Record
            record = not (idx % record_interval)
            last_step = (idx==steps-1)
            if record or last_step:
                if last_step:
                    a_t_record[-1,:] = pulse_out.a_t
                    a_v_record[-1,:] = pulse_out.a_v
                    z_record[-1] = z
                elif record:
                    a_t_record[idx//record_interval,:] = pulse_out.a_t
                    a_v_record[idx//record_interval,:] = pulse_out.a_v
                    z_record[idx//record_interval] = z

                #---- Plot
                if plot is not None:
                    # Update Plots
                    self.update_plots(plot, pulse_out, z)
                    # End animation with the last step
                    if last_step:
                        for artist in self._artists:
                            artist.set_animated(False)

        return pulse_out, a_t_record, a_v_record, z_record

    def propagate(self, a_v, z_start, z_stop, dz, local_error, k5_v=None):
        '''Propagates from `z_start` to `z_stop` using an adaptive step size
        algorithm based on an embedded 4th order Runge-Kutta method in the
        interaction picture (ERK4(3)IP) [1].

        The local error is estimated as the difference between the 3rd and 4th
        order result.


        [1]: S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for step-size
        control in the interaction picture method," Computer Physics
        Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

        https://doi.org/10.1016/j.cpc.2012.12.020
        '''
        z = z_start
        while z < z_stop:
            z_next = z + dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz_next = dz
                dz = z_next - z
            else:
                final_step = False

            #---- Integrate by dz
            a_RK4_v, a_RK3_v, k5_v_next = self.integrate(
                a_v,
                z,
                dz,
                k5_v=k5_v)

            #---- Estimate Relative Local Error
            L2_norm = np.sum(a_RK4_v.real**2 + a_RK4_v.imag**2)**0.5
            rel_error = (a_RK4_v - a_RK3_v)/L2_norm
            L2_error = np.sum(rel_error.real**2 + rel_error.imag**2)**0.5
            error_ratio = (L2_error/local_error)**0.25
            #print('{:.6g},\t{:.2g},\t{:.2g}'.format(z, dz, error_ratio))

            #---- Propagate Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                dz = dz/2
            else:
                # Update the integration parameters
                z = z_next
                a_v = a_RK4_v
                if (not final_step) or (error_ratio > 1):
                    # Update the step size
                    dz_next = dz / max(error_ratio, 0.5)
        return a_v, z_next, dz_next, k5_v_next

    def integrate(self, a_v, z, dz, k5_v=None):
        """Integrates over a step size of `dz` using an embedded 4th order
        Runge-Kutta method in the interaction picture (ERK4(3)IP) based on [1].

        This method contains an embedded Runge–Kutta scheme with orders 3 and
        4, which allows for an estimate of the local error.


        [1]: S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for step-size
        control in the interaction picture method," Computer Physics
        Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

        https://doi.org/10.1016/j.cpc.2012.12.020
        """
        #---- k1
        if k5_v is None:
            self.update_linearity(z)
            self.update_nonlinearity(z)
            k5_v = self.nonlinear(a_v)
        IP_in_op_v = self.linear_operator(0.5*dz) # interaction picture
        aI_v = IP_in_op_v * a_v
        kI1_v = IP_in_op_v * k5_v

        #---- k2
        self.update_nonlinearity(z+0.5*dz)
        aI2_v = aI_v + 0.5*dz * kI1_v
        kI2_v = self.nonlinear(aI2_v)

        #---- k3
        aI3_v = aI_v + 0.5*dz * kI2_v
        kI3_v = self.nonlinear(aI3_v)

        #---- k4
        self.update_linearity(z+dz)
        IP_out_op_v = self.linear_operator(0.5*dz)

        self.update_nonlinearity(z+dz)
        aI4_v = aI_v + dz*kI3_v
        a4_v = IP_out_op_v * aI4_v
        k4_v = self.nonlinear(a4_v)

        #---- RK4
        bI_v = aI_v + dz/6 * (kI1_v + 2*(kI2_v + kI3_v))
        b_v = IP_out_op_v * bI_v
        a_RK4_v = b_v + dz/6 * k4_v

        #---- k5
        k5_v = self.nonlinear(a_RK4_v)

        #---- RK3
        a_RK3_v = b_v + dz/30 * (2*k4_v + 3*k5_v)

        return a_RK4_v, a_RK3_v, k5_v

    #---- Z-Dependent Methods
    def update_linearity(self, z):
        '''Updates all z-dependent linear parameters'''
        #---- Gain
        if self._update_alpha:
            self.alpha = self.mode._alpha(z)

        #---- Phase
        if self._update_beta:
            self.beta = self.mode._beta(z)
            self.beta1_v0 = fdd(self.beta, self.dw, self.v0_idx)
            # Beta in comoving frame
            self.beta_cm = self.beta - self.beta1_v0*self.w_grid

        #---- Propagation Constant
        if self._update_kappa:
            if self.alpha is not None:
                self.kappa_cm = self.beta_cm + 0.5j*self.alpha
            else:
                self.kappa_cm = self.beta_cm

    def update_nonlinearity(self, z):
        '''Updates all z-dependent nonlinear parameters'''
        #---- 2nd Order
        if self._update_g2:
            self.g2 = self.mode._g2(z)

        #---- 3rd Order
        if self._update_g3:
            self.g3 = self.mode._g3(z)
        if self._update_r3:
            self.r3 = self.mode._r3(z)

    #---- Operators
    def linear_operator(self, dz):
        '''Returns the linear operator in the frequency domain.'''
        # Linear Operator
        l_v = -1j*dz * self.kappa_cm
        return exp(l_v)

    def nonlinear(self, a_v):
        '''Returns the nonlinear operator in the frequency domain.'''
        self._nl_v[:] = 0j

        #---- Carrier-resolved Field
        self._nl_a_v[self.cr_pre_slice] = 0j
        self._nl_a_v[self.cr_slice] = a_v
        self._nl_a_v[self.cr_post_slice] = 0j

        nl_a_t = fft.irfft(self._nl_a_v, fsc=self.nl_dt * 2**0.5, n=self.nl_n_points)
        nl_a2_t = nl_a_t * nl_a_t

        #---- 2nd Order Nonlienarity
        if self.g2 is not None:
            nl2_a2_v = fft.rfft(nl_a2_t, fsc=self.nl_dt * 2**0.5)
            a2_v = nl2_a2_v[self.cr_slice]
            self._nl_v -= self.g2 * a2_v

        #---- 3rd Order Nonlienarity
        if self.g3 is not None:
            if self.r3 is not None:
                nl3_a2_v = (fft.rfft(nl_a2_t, fsc=self.nl_dt) if self.g2 is None
                            else nl2_a2_v * 2**-0.5)
                nl3_a2r3_v = nl3_a2_v * self.r3
                nl3_a2r3_t = fft.irfft(nl3_a2r3_v, fsc=self.nl_dt, n=self.nl_n_points)
            else:
                nl3_a2r3_t = nl_a2_t
            nl3_a3r3_t = nl_a_t * nl3_a2r3_t
            nl3_a3r3_v = fft.rfft(nl3_a3r3_t, fsc=self.nl_dt * 2**0.5)
            a3r3_v = nl3_a3r3_v[self.cr_slice]
            self._nl_v -= self.g3 * a3r3_v

        #---- Nonlinear Response
        return self._1j_w_grid * self._nl_v # "-" included in nl_v

    #---- Plot Methods
    def setup_plots(self, plot, pulse_out, z):
        '''Initializes figure for real-time plotting of a simulation'''
        self._rt_fig = self._plt.figure("Real-Time Simulation", clear=True)
        self._ax_0 = self._plt.subplot2grid((2,1), (0,0), fig=self._rt_fig)
        self._ax_1 = self._plt.subplot2grid((2,1), (1,0), sharex=self._ax_0, fig=self._rt_fig)

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

        # Z Label
        self._z_label = self._ax_1.legend(
            [],[],
            title='z = {:.6g} m'.format(z),
            loc=9,
            labelspacing=0,
            framealpha=1,
            shadow=False)
        self._z_label.set_animated(True)

        # Layout
        self._rt_fig.tight_layout()
        self._rt_fig.canvas.draw()

        # Setup Blit
        self._artists = (self._ln_pwr, self._ln_phs, self._z_label)

        self._rt_fig_bkg_0 = self._rt_fig.canvas.copy_from_bbox(self._ax_0.bbox)
        self._rt_fig_bkg_1 = self._rt_fig.canvas.copy_from_bbox(self._ax_1.bbox)

    def update_plots(self, plot, pulse_out, z):
        '''Updates figure for real-time plotting during a simulation'''
        # Restore Background
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_0)
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_1)

        # Update Data
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

        # Update Z Label
        self._z_label.set_title('z = {:.6g} m'.format(z))

        # Blit
        for artist in self._artists:
            artist.axes.draw_artist(artist)

        self._rt_fig.canvas.blit(self._ax_0.bbox)
        self._rt_fig.canvas.blit(self._ax_1.bbox)
        self._rt_fig.canvas.start_event_loop(1e-6)
