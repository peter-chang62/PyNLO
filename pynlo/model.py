# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = ["UPE"]

import numpy as np
from scipy.constants import c, pi
from numba import njit

from pynlo import light, media
from pynlo.utility import fft


# %% Routines

@njit(parallel=True)
def exp(x):
    return np.exp(x)

@njit(parallel=True)
def add(x, y):
    return x + y

@njit(parallel=True)
def mult(x, y):
    return x * y

@njit(parallel=True)
def prw(x, y):
    return x**y


# %% Classes

class UPE():
    def __init__(self, pulse, mode, antialias=False):
        """
        ...single mode unidirectional propagation equation...
        both 3rd and 2nd order nonlinearities...

        Parameters
        ----------
        pulse : pynlo.light.Pulse
            DESCRIPTION.
        mode : pynlo.media.Mode
            DESCRIPTION.
        antialias : TYPE, optional
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
        assert isinstance(pulse, light.Pulse)
        assert isinstance(mode, media.Mode)

        self.pulse0 = pulse
        self.mode = mode

        # Absolute Angular Frequency
        self.w_grid = 2*pi*self.v_grid
        self.w_ref = 2*pi*self.v_ref

        # Wavelength
        self.wl_grid = c/self.v_grid
        self.dJdHz_to_dJdm = (self.v_grid**2/c) # density conversion factor

        # Temporal Grid
        self.dt = 1/(self.n_points * self.dv)
        self.t_grid = self.dt*(np.arange(self.n_points) - (self.n_points//2))
        self.t_window = self.dt * self.n_points

        #--- Waveguide Parameters
        self.alpha = alpha
        self.beta = beta
        self.beta1_0 = beta1_0 # 1/(group velocity) of the retarted frame
        self.g2 = g2
        self.g3 = g3

        #--- Carrier Resolved Grid
        cr_v_min_idx = int(round(self.v_grid.min()/self.dv))
        cr_v_max_idx = int(round(self.v_grid.max()/self.dv))
        self.cr_idx = np.array([cr_v_min_idx, cr_v_max_idx+1])
        self.cr_slice = slice(self.cr_idx[0], self.cr_idx[1])
        self.cr_pre_slice = slice(None, self.cr_idx[0])
        self.cr_post_slice = slice(self.cr_idx[1], None)

        self.cr_n_points = next_fast_len(2*cr_v_max_idx - 1)

        # Temporal Grid
        self.cr_dt = 1/(self.cr_n_points * self.dv)
        self.cr_t_grid = self.cr_dt*(np.arange(self.cr_n_points) - (self.cr_n_points//2))

        # Real FFT Absolute Frequency Grid
        self.cr_v_grid = self.dv*np.arange(self.cr_n_points//2 + 1)

        #--- Nonlinear Time and Frequency Grid
        if antialias == False:
            # 1x points with aliasing
            self.nl_n_points = self.cr_n_points
        elif self.g3 is not None:
            # 3x points to eliminate aliasing
            self.nl_n_points = next_fast_len(3*self.cr_n_points)
        elif self.g2 is not None:
            # 2x points to eliminate aliasing
            self.nl_n_points = next_fast_len(2*self.cr_n_points)

        # Temporal Grid
        self.nl_dt = 1/(self.nl_n_points*self.dv)
        self.nl_t_grid = self.nl_dt*(np.arange(self.nl_n_points) - (self.nl_n_points//2))

        # Real FFT Absolute Frequency Grid
        self.nl_v_grid = self.dv*np.arange(self.nl_n_points//2 + 1)

        #--- Nonlinear Parameters
        self.chi2 = True if self.g2 is not None else False
        self.chi3 = True if self.g3 is not None else False
        self.dispersive_chi2 = dispersive_chi2
        self.dispersive_chi3 = dispersive_chi3
        self.R2_v, self.R3_v = self.calculate_nonlinear_susceptibility()
        self.nl_ref_exp_phase = exp(+1j*(self.w_ref * self.nl_t_grid))

        self.nl_v = np.zeros_like(self.v_grid, dtype=complex)
        self.nl_a_v = np.zeros_like(self.nl_v_grid, dtype=complex)

        #--- Optical Pulse
        # Frequency Domain
        self.A0_v = np.asarray(A_v)
        self.A_v = self.A0_v.copy()

        # Time Domain
        self.A0_t = fftshift(mkl_fft.ifft(ifftshift(self.A0_v) * (self.dv*self.n_points)))
        self.A_t = self.A0_t.copy()


    def simulate(self,
        z_grid, dz=None, local_error=1e-6, reset_A=True,
        plot_time=False, plot_frq=False, plot_wvl=False,
        record_interval=1):
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
        reset_A : float, optional
            Determines whether to use the initial pulse or the most recently
            simulated value. The default is ``True``, which uses the initial
            pulse.
        plot_time : bool, optional
        plot_frq : bool, optional
        plot_wvl : bool, optional
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
        A_t_record : array of complex
            The root-power complex envelope of the output pulses.
        v_grid : array of float
            The frequency grid of the output pulses.
        A_v_record
            The root-power spectrum of the output pulses.
        z_record
            The points along the waveguide at which the pulses were returned.
        """
        #--- Z Sample Space
        self.z_grid = np.asarray(z_grid)
        self.z = self.z_grid[0]
        n_records = (self.z_grid.size-2)//record_interval + 2
        self.z_record = np.empty((n_records,), dtype=np.complex)
        self.z_record[0] = self.z

        #--- Optical Pulse
        # Frequency Domain
        if reset_A:
            self.A_v = self.A0_v.copy()
        self.A_v_record = np.empty((n_records,len(self.A_v)), dtype=np.complex)
        self.A_v_record[0,:] = self.A_v
        # Time Domain
        if reset_A:
            self.A_t = self.A0_t.copy()
        self.A_t_record = np.empty((n_records,len(self.A_t)), dtype=np.complex)
        self.A_t_record[0,:] = self.A_t

        #--- Plotting
        self.plot_time = plot_time
        self.plot_frq = plot_frq
        self.plot_wvl = plot_wvl
        self.plotting = plot_time + plot_frq + plot_wvl
        assert self.plotting <= 1
        if self.plotting:
            # Import if needed
            try:
                self.plt
            except AttributeError:
                import matplotlib.pyplot as plt
                self.plt = plt
            # Setup Plots
            self.setup_plots()

        #--- Step Size
        self.local_error = local_error
        if dz is None:
            dz = self.estimate_step_size(self.A_v, self.z, self.local_error)
            print("Initial Step Size:\t{:.3g}".format(dz))
        self.dz = dz

        #--- Propagate
        k5_v = None
        steps = self.z_grid.size
        for idx in range(1, steps):
            #--- Propagate
            z_stop = self.z_grid[idx]
            (self.A_v, self.z, k5_v) = self.propagate(
                self.A_v,
                self.z,
                z_stop,
                k5_v=k5_v)

            #--- Record and Plot
            record = not(idx % record_interval)
            last_step = (idx==steps-1)
            if record or last_step:
                self.A_t = fftshift(mkl_fft.ifft(ifftshift(self.A_v) * (self.dv*self.n_points), overwrite_x=True))

                #--- Record
                if last_step:
                    self.A_t_record[-1,:] = self.A_t
                    self.A_v_record[-1,:] = self.A_v
                    self.z_record[-1] = self.z
                elif record:
                    self.A_t_record[idx//record_interval,:] = self.A_t
                    self.A_v_record[idx//record_interval,:] = self.A_v
                    self.z_record[idx//record_interval] = self.z

                #--- Plot
                if self.plotting:
                    # Update Plots
                    self.update_plots()
                    # End animation with the last step
                    if last_step:
                        for artist in self._artists:
                            artist.set_animated(False)

        return self.t_grid, self.A_t_record, self.v_grid, self.A_v_record, self.z_record


    def propagate(self, A_v, z_start, z_stop, k5_v=None):
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
            dz = self.dz
            z_next = z + dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz = z_next - z
            else:
                final_step = False

            #--- Integrate by dz
            A_RK4_v, A_RK3_v, k5_next_v = self.integrate(
                A_v,
                z,
                dz,
                k5_v=k5_v)

            #--- Estimate the Relative Local Error
            L2_norm = np.sum(A_RK4_v.real**2 + A_RK4_v.imag**2)**0.5
            rel_error = (A_RK4_v - A_RK3_v)/L2_norm
            L2_error = np.sum(rel_error.real**2 + rel_error.imag**2)**0.5
            error_ratio = (L2_error/self.local_error)**0.25
            #print('{:.6g},\t{:.2g},\t{:.2g}'.format(z, dz, error_ratio))

            #--- Propagate the Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                self.dz = dz/2
                continue
            else:
                # Update the integration parameters
                z = z_next
                A_v = A_RK4_v
                k5_v = k5_next_v
                if (not final_step) or (error_ratio > 1):
                    # Update the step size
                    self.dz = dz / max(error_ratio, 0.5)
        return A_v, z, k5_v


    def integrate(self, A_v, z, dz, k5_v=None):
        """Integrates over a step size of `dz` using an embedded 4th order
        Runge-Kutta method in the interaction picture (ERK4(3)IP) based on [1].

        This method contains an embedded Runge–Kutta scheme with orders 3 and
        4, which allows for an estimate of the local error.


        [1]: S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for step-size
        control in the interaction picture method," Computer Physics
        Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

        https://doi.org/10.1016/j.cpc.2012.12.020
        """

        IP_in_op_v = self.linear_operator(z, 0.5*dz)
        # alpha, beta, beta1_0 = self.effective_linearity(z)
        # IP_in_op_v = linear_operator(self.mode, 0.5*dz, alpha, beta, beta1_0)

        IP_out_op_v = self.linear_operator(z+dz, 0.5*dz)
        # alpha, beta, beta1_0 = self.effective_linearity(z+dz)
        # IP_out_op_v = linear_operator(self.mode, 0.5*dz, alpha, beta, beta1_0)

        #--- Interaction Picture
        AI_v = IP_in_op_v * A_v

        #--- k1
        if k5_v is None:
            g2_v, g3_v, R2_v, R3_v = self.effective_nonlinearity(z)
            k5_v = self.nonlinear(A_v, g2_v, g3_v, R2_v, R3_v)
            # k5_v = nonlinear(self.mode, A_v, g2_v, g3_v, R2_v, R3_v)
        kI1_v = IP_in_op_v * k5_v

        #--- k2
        g2_v, g3_v, R2_v, R3_v = self.effective_nonlinearity(z+0.5*dz)
        AI2_v = AI_v + (0.5*dz)*kI1_v
        kI2_v = self.nonlinear(AI2_v, g2_v, g3_v, R2_v, R3_v)
        # kI2_v = nonlinear(self.mode, AI2_v, g2_v, g3_v, R2_v, R3_v)

        #--- k3
        AI3_v = AI_v + (0.5*dz)*kI2_v
        kI3_v = self.nonlinear(AI3_v, g2_v, g3_v, R2_v, R3_v)
        # kI3_v = nonlinear(self.mode, AI3_v, g2_v, g3_v, R2_v, R3_v)

        #--- k4
        g2_v, g3_v, R2_v, R3_v = self.effective_nonlinearity(z+dz)
        AI4_v = AI_v + dz*kI3_v
        A4_v = IP_out_op_v * AI4_v
        k4_v = self.nonlinear(A4_v, g2_v, g3_v, R2_v, R3_v)
        # k4_v = nonlinear(self.mode, A4_v, g2_v, g3_v, R2_v, R3_v)

        #--- RK4
        bI_v = AI_v + (dz/6)*(kI1_v + 2*(kI2_v + kI3_v))
        b_v = IP_out_op_v * bI_v
        A_RK4_v = b_v + dz*k4_v/6

        #--- k5
        k5_v = self.nonlinear(A_RK4_v, g2_v, g3_v, R2_v, R3_v)
        # k5_v = nonlinear(self.mode, A_RK4_v, g2_v, g3_v, R2_v, R3_v)

        #--- RK3
        A_RK3_v = b_v + (dz/30)*(2*k4_v + 3*k5_v)

        return A_RK4_v, A_RK3_v, k5_v


    def effective_linearity(self, z):
        # Loss
        alpha = self.alpha(z=z)

        # Phase
        beta = self.beta(z=z)
        beta1_0 = self.beta1_0(z=z)
        return alpha, beta, beta1_0


    def effective_nonlinearity(self, z):
        g2_v = self.g2(z=z) if self.chi2 else 0.
        g3_v = self.g3(z=z) if self.chi3 else 0.
        R2_v = self.R2_v if self.dispersive_chi2 else 0.
        R3_v = self.R3_v if self.dispersive_chi3 else 0.

        return g2_v, g3_v, R2_v, R3_v


    def linear_operator(self, z, dz):
        '''Returns the linear operator in the frequency domain.'''
        alpha, beta, beta1_0 = self.effective_linearity(z)

        # Beta in comoving frame
        beta_cm = beta - beta1_0*self.w_grid

        # Linear Operator
        l_v = (alpha/2 - 1j*beta_cm) * dz
        exp_l_v = exp(l_v)
        return exp_l_v


    def nonlinear(self, A_v, g2_v, g3_v, R2_v, R3_v):
        '''Returns the result of the nonlinear operator in the frequency
        domain.
        '''
        self.nl_v[:] = 0

        #--- Carrier-resolved Field
        #--------------------------
        self.nl_a_v[self.cr_pre_slice] = 0
        self.nl_a_v[self.cr_slice] = 2**-0.5 * A_v
        self.nl_a_v[self.cr_post_slice] = 0

        nl_a_v = self.nl_a_v
        nl_a_t = mkl_fft.irfft_numpy(
            nl_a_v * (self.dv*self.nl_n_points),
            n=self.nl_n_points)

        #--- Chi2
        #--------
        if self.chi2:
            if self.dispersive_chi2:
                nl2_aR2_v = nl_a_v * R2_v
                nl2_aR2_t = mkl_fft.irfft_numpy(
                    nl2_aR2_v * (self.dv*self.nl_n_points),
                    n=self.nl_n_points)
            else:
                nl2_aR2_t = nl_a_t

            nl2_a2R2_t = nl_a_t * nl2_aR2_t
            nl2_a2R2_v = mkl_fft.rfft_numpy(
                nl2_a2R2_t * self.nl_dt)

            A2R2_v = 2**0.5 * nl2_a2R2_v[self.cr_slice]

            self.nl_v += g2_v * A2R2_v

        #--- Chi3
        #--------
        if self.chi3:
            nl3_a2_t = nl_a_t**2
            if self.dispersive_chi3:
                nl3_a2_v = mkl_fft.rfft_numpy(
                    nl3_a2_t * self.nl_dt)
                nl3_a2R3_v = nl3_a2_v * R3_v
                nl3_a2R3_t = mkl_fft.irfft_numpy(
                    nl3_a2R3_v * (self.dv*self.nl_n_points),
                    n=self.nl_n_points)
            else:
                nl3_a2R3_t = nl3_a2_t

            nl3_a3R3_t = nl_a_t * nl3_a2R3_t
            nl3_a3R3_v = mkl_fft.rfft_numpy(
                nl3_a3R3_t * self.nl_dt)

            A3R3_v = 2**0.5 * nl3_a3R3_v[self.cr_slice]

            self.nl_v += g3_v * A3R3_v

        #--- Nonlinear Response
        #----------------------
        return -1j * self.w_grid * self.nl_v


    def calculate_nonlinear_susceptibility(self):
        '''Returns the RFFT of the normalized nonlinear susceptibilities chi2
        and chi3 in the frequency domain.

        The returned values are normalized such that sum(R*dt) = 1. Common
        functions are the instantaneous (dirac delta) and raman responses.

        Notes
        -----
        These relations only contain the nonlinear dispersion of the bulk
        material responses. Nonlinear dispersion attributable to the waveguide
        mode should be included in the g2 and g3 parameters.
        '''
        def dirac_delta():
            dd_t = np.zeros_like(self.nl_t_grid)
            dd_t[0] = 1/self.nl_dt
            dd_t = fftshift(dd_t)
            return dd_t

        def R_a(tau1, tau2, fraction):
            t_delay = self.nl_t_grid
            RT = ((tau1**2+tau2**2)/(tau1*tau2**2))*exp(t_delay/tau2)*np.sin(t_delay/tau1)
            RT[t_delay > 0] = 0
            RT *= fraction
            return RT

        #--- chi2
        # Instantaneous
        R2_v = mkl_fft.rfft_numpy(ifftshift(dirac_delta()) * self.nl_dt)

        #--- chi3
        # Instantaneous
        R3_v = mkl_fft.rfft_numpy(ifftshift(dirac_delta()) * self.nl_dt)

        # PPLN Raman Response
        # see arXiv:1211.1721 for coefficients
        raman_fraction = 0.2
        #           tau1        tau2        fraction
        weights = [[21e-15,     544e-15,    0.635],
                   [19.3e-15,   1021e-15,   0.105],
                   [15.9e-15,   1361e-15,   0.020],
                   [8.3e-15,    544e-15,    0.240]]

        raman_t = np.zeros_like(self.nl_t_grid)
        for weight in weights:
            raman_t += R_a(*weight)

        raman_t *= 1/np.sum(raman_t * self.nl_dt)
        raman_v = mkl_fft.rfft_numpy(ifftshift(raman_t) * self.nl_dt)

        R3_v = (1.-raman_fraction)*R3_v + raman_fraction*raman_v

        return R2_v, R3_v


    def pulse_energy(self):
        return np.sum(self.A_v.real**2 + self.A_v.imag**2) * self.dv


    def estimate_step_size(self, A_v, z, local_error=1e-6, dz=1e-5):
        '''Estimate the optimal step size after integrating by a test dz'''
        #--- Integrate by dz
        A_RK4_v, A_RK3_v, k5_next_v = self.integrate(
            A_v,
            z,
            dz)

        #--- Estimate the Relative Local Error
        L2_norm = np.sum(A_RK4_v.real**2 + A_RK4_v.imag**2)**0.5
        rel_error = (A_RK4_v - A_RK3_v)/L2_norm
        L2_error = np.sum(rel_error.real**2 + rel_error.imag**2)**0.5
        error_ratio = (L2_error/local_error)**0.25
        #print('{:.3g}'.format(error_ratio))

        dz = dz * 0.5*(1 + 1/min(2, max(error_ratio, 0.5))) # approach the optimum step size
        return dz


    def setup_plots(self):
        '''The initialization function called at the start of plotting'''
        self._rt_fig = self.plt.figure("Real Time Simulation", clear=True)
        self._ax_0 = self.plt.subplot2grid((2,1), (0,0), fig=self._rt_fig)
        self._ax_1 = self.plt.subplot2grid((2,1), (1,0), sharex=self._ax_0, fig=self._rt_fig)

        if self.plot_time:
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e12*self.t_grid,
                self.A_t.real**2+self.A_t.imag**2,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e12*self.t_grid,
                1e-12*(self.v_ref+np.gradient(np.unwrap(np.angle(self.A_t))/(2*pi), self.dt)),
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

        if self.plot_frq:
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e-12*self.v_grid,
                self.A_v.real**2+self.A_v.imag**2,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e-12*self.v_grid,
                -1e12*np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.dv),
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

        if self.plot_wvl:
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e9*self.wl_grid,
                self.dJdHz_to_dJdm * (self.A_v.real**2+self.A_v.imag**2),
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e9*self.wl_grid,
                -1e12*np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.dv),
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
            title='z = {:.9g} m'.format(self.z),
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


    def update_plots(self):
        # Restore Background
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_0)
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_1)

        # Update Data
        if self.plot_time:
            self._ln_pwr.set_data(
                1e12*self.t_grid,
                self.A_t.real**2+self.A_t.imag**2)
            self._ln_phs.set_data(
                1e12*self.t_grid,
                1e-12*(self.v_ref + np.gradient(np.unwrap(np.angle(self.A_t))/(2*pi), self.dt)))

        if self.plot_frq:
            self._ln_pwr.set_data(
                1e-12*self.v_grid,
                self.A_v.real**2+self.A_v.imag**2)
            self._ln_phs.set_data(
                1e-12*self.v_grid,
                -1e12*np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.dv))

        if self.plot_wvl:
            self._ln_pwr.set_data(
                1e9*self.wl_grid,
                self.dJdHz_to_dJdm * (self.A_v.real**2+self.A_v.imag**2))
            self._ln_phs.set_data(
                1e9*self.wl_grid,
                -1e12*np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.dv))

        # Update Z Label
        self._z_label.set_title('z = {:.6g} m'.format(self.z))

        # Blit
        for artist in self._artists:
            artist.axes.draw_artist(artist)

        self._rt_fig.canvas.blit(self._ax_0.bbox)
        self._rt_fig.canvas.blit(self._ax_1.bbox)
        self._rt_fig.canvas.start_event_loop(1e-6)

