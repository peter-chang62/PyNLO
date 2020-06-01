# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = ["nlee"]

import numpy as np
from scipy.fftpack import fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len
import mkl_fft
from scipy.constants import c, pi
from numba import njit


# %% Routines

@njit(parallel=True)
def resample_spectrum(spectrum, new_size):
    '''Zero pad or truncates the high frequency components of an fft signal to
    add or remove frequency content.

    This assumes that the input is ordered in the same way as returned by
    `scipy.fftpack.fft`.
    '''
    old_spectrum = np.asarray(spectrum)

    # Initialize New Array
    old_size = old_spectrum.size
    if new_size > old_size:
        new_spectrum = np.zeros(new_size, dtype=old_spectrum.dtype)
    else:
        new_spectrum = np.empty(new_size, dtype=old_spectrum.dtype)

    size = min((old_size, new_size))
    # Positive Frequencies
    p_frq = (size // 2) + (size % 2)
    # Negative Frequencies
    n_frq = -(size // 2)
    # New Spectrum
    new_spectrum[:p_frq] = old_spectrum[:p_frq]
    new_spectrum[n_frq:] = old_spectrum[n_frq:]
    return new_spectrum

@njit(parallel=True)
def rfft_to_analytic(R_v, n_points, slice_real, slice_imag):
    '''Converts an rfft into its analytic fft form. The L2 norm of this
    transform is 2 times the L2 norm of the original.

    This assumes that the input is ordered in the same way as returned by
    `scipy.fftpack.rfft`. The slices set the frequency components to be placed
    in the new array. The slices must completely fill the new array.
    '''
    A_v = np.empty(n_points, dtype=np.complex128)
    A_v.real = R_v[slice_real]
    A_v.imag = R_v[slice_imag]
    A_v *= 2
    return A_v

@njit(parallel=True)
def analytic_to_rfft(A_v, n_points, slice_real, slice_imag):
    '''Converts the positive frequencies of an analytic fft into the rfft form.
    The L2 norm of this transform is 0.5 times the L2 norm of the original.

    This assumes that the input is ordered in the same way as returned by
    `scipy.fftpack.fft`, up until the negative frequencies. The slices
    set the location of the frequency components in the new array.
    '''
    A_v = A_v*0.5
    R_v = np.zeros(n_points, dtype=np.float64)
    R_v[slice_real] = A_v.real
    R_v[slice_imag] = A_v.imag
    return R_v

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

def gaussian_window(x, x0, fwhm):
    dx = np.gradient(x, axis=0)
    sig = 0.5*fwhm/(2*np.log(2))**0.5
    window = np.exp(-(x-x0)**2/(2*sig**2))
    window /= np.sum(window*dx, axis=0, keepdims=True)
    return window

# %% Classes

class SM_UPE():
    def __init__(self,
        v_grid, df, A_v,
        alpha, beta, beta1, gamma2, gamma3,
        ref_v=None,
        dispersive_chi2=False, dispersive_chi3=False):
        '''

        '''
        #--- Linear Time and Frequency Grid
        self.n_points = len(v_grid)
        # Absolute Frequency
        self.v_grid = np.asarray(v_grid)
        self.df = df
        if ref_v is None:
            ref_v = ifftshift(self.v_grid)[0]
        self.ref_v_idx = ((self.v_grid-ref_v)**2).argmin()
        self.ref_v = self.v_grid[self.ref_v_idx]
        # Check Frequency Grid Alignment
        v_grid_zero_offset = np.modf(self.v_grid.min() / self.df)
        v_grid_zero_test_offset = np.modf((v_grid_zero_offset[1]*df)/df)
        if v_grid_zero_offset != v_grid_zero_test_offset:
            v_grid_zero_offset = v_grid_zero_offset[0] * self.df
            if v_grid_zero_offset > self.df/2:
                v_grid_zero_offset -= self.df
            if v_grid_zero_offset != 0:
                print("Found a frequency grid offset of {:.3g} Hz (fractional offset of {:.3g}).".format(v_grid_zero_offset, v_grid_zero_offset/self.df),
                    "The frequency grid must be aligned with 0 Hz for proper function.",
                    sep="\n")
                assert v_grid_zero_offset==0
        # Relative Frequency
        self.f_grid = self.v_grid - self.ref_v
        # Absolute Angular Frequency
        self.w_grid = 2*pi*self.v_grid
        self.ref_w = 2*pi*self.ref_v
        # Relative Angular Frequency
        self.af_grid = 2*pi*self.f_grid
        # Wavelength
        self.wl_grid = c/self.v_grid
        self.dJdHz_to_dJdm = (self.v_grid**2/c) # density conversion factor
        # Temporal Grid
        self.t_grid = fftshift(fftfreq(self.n_points, d=self.df)) #[-dt*N/2, +dt*N/2)
        self.dt = np.diff(self.t_grid).mean()
        self.t_window = self.t_grid.max() - self.t_grid.min()

        #--- Waveguide Parameters
        self.alpha = alpha
        self.beta = beta
        self.beta1 = beta1 # 1/(group velocity) of the retarted frame
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        #--- Carrier Resolved Grid
        self.n_min_offset = int(round(self.v_grid.min()/self.df))
        self.n_max_offset = int(round(self.v_grid.max()/self.df))
        self.idx_start_real = self.n_min_offset*2-1
        self.idx_stop_real = self.n_max_offset*2
        self.idx_start_imag = self.idx_start_real + 1
        self.idx_stop_imag = self.idx_stop_real + 1
        self.slice_real = slice(self.idx_start_real,self.idx_stop_real,2)
        self.slice_imag = slice(self.idx_start_imag,self.idx_stop_imag,2)
        self.cr_n_points = next_fast_len(2*self.n_max_offset)

        # Temporal Grid
        self.cr_t_grid = fftshift(fftfreq(self.cr_n_points, d=self.df))
        self.cr_dt = np.diff(self.cr_t_grid).mean()
        self.cr_blackman = np.blackman(self.cr_n_points)
        # Relative Frequency Grid
        self.cr_f_grid = fftshift(fftfreq(self.cr_n_points, d=self.cr_dt))
        self.cr_df = np.diff(self.cr_f_grid).mean()
        # Real FFT Absolute Frequency Grid
        self.cr_rv_grid = rfftfreq(self.cr_n_points, d=self.cr_dt)

        #--- Nonlinear Time and Frequency Grid
        if (self.gamma2 is not None) or (self.gamma3 is not None):
            if self.gamma3 is not None:
                # 4x points to eliminate aliasing
                self.nl_n_points = next_fast_len(4*(self.n_points+self.n_min_offset))
            elif self.gamma2 is not None:
                # 3x points to eliminate aliasing
                self.nl_n_points = next_fast_len(3*(self.n_points+self.n_min_offset))
            # Temporal Grid
            self.nl_t_grid = fftshift(fftfreq(self.nl_n_points, d=self.df))
            self.nl_dt = np.diff(self.nl_t_grid).mean()
            self.nl_blackman = np.blackman(self.nl_n_points)
            # Relative Frequency Grid
            self.nl_f_grid = fftshift(fftfreq(self.nl_n_points, d=self.nl_dt))
            self.nl_df = np.diff(self.nl_f_grid).mean()
            # Real FFT Absolute Frequency Grid
            self.nl_rv_grid = rfftfreq(self.nl_n_points, d=self.nl_dt)

            #--- Nonlinear Parameters
            self.dispersive_chi2=dispersive_chi2
            self.dispersive_chi3=dispersive_chi3
            self.R2_v, self.R3_v = self.calculate_nonlinear_susceptibility()
            #self.nl_ref_exp_phase = exp(+1j*(self.ref_w * self.nl_t_grid))

        #--- Optical Pulse
        # Frequency Domain
        self.A0_v = np.asarray(A_v)
        self.A_v = self.A0_v.copy()
        # Time Domain
        self.A0_t = mkl_fft.ifft(ifftshift(self.A0_v) * self.df*self.n_points)
        self.A_t = self.A0_t.copy()


    def simulate(self,
        z_grid, dz=None, local_error=1e-6, reset_A=True,
        plot_time=False, plot_frq=False, plot_wvl=False,
        save_interval=1):
        '''

        '''
        #--- Z Sample Space
        self.z_grid = np.asarray(z_grid)
        self.z = self.z_grid[0]
        n_records = (self.z_grid.size-2)//save_interval + 2
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
            self.A_t = mkl_fft.ifft(ifftshift(self.A0_v) * self.df*self.n_points)
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
            print(dz)
        self.dz = dz

        #--- Propagate
        k5_v = None
        ref_phase = 0
        steps = self.z_grid.size
        for idx in range(1, steps):
            #--- Propagate
            z_stop = self.z_grid[idx]
            (self.A_v, self.z, k5_v, ref_phase) = self.propagate(
                self.A_v,
                self.z,
                z_stop,
                k5_v=k5_v,
                ref_phase=ref_phase)

            #--- Record and Plot
            record = not(idx % save_interval)
            last_step = (idx==steps-1)
            if record or last_step:
                self.A_t = mkl_fft.ifft(ifftshift(self.A_v) * self.df*self.n_points, overwrite_x=True)

                #--- Record
                if last_step:
                    self.A_t_record[-1,:] = self.A_t
                    self.A_v_record[-1,:] = self.A_v
                    self.z_record[-1] = self.z
                elif record:
                    self.A_t_record[idx//save_interval,:] = self.A_t
                    self.A_v_record[idx//save_interval,:] = self.A_v
                    self.z_record[idx//save_interval] = self.z

                #--- Plot
                if self.plotting:
                    # Update Plots
                    self.update_plots()
                    # End animation with the last step
                    if last_step:
                        for artist in self._artists:
                            artist.set_animated(False)

        return self.t_grid, self.A_t_record, self.v_grid, self.A_v_record, self.z_record


    def propagate(self, A_v, z_start, z_stop, k5_v=None, ref_phase=0):
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
            A_RK4_v, A_RK3_v, k5_next_v, ref_phase_end = self.integrate(
                A_v,
                z,
                dz,
                k5_v=k5_v,
                ref_phase=ref_phase)

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
                ref_phase = ref_phase_end
                if (not final_step) or (error_ratio > 1):
                    # Update the step size
                    self.dz = dz / max(error_ratio, 0.5)
        return A_v, z, k5_v, ref_phase


    def integrate(self, A_v, z, dz, k5_v=None, ref_phase=0):
        """Integrates over a step size of `dz` using an embedded 4th order
        Runge-Kutta method in the interaction picture (ERK4(3)IP) based on [1].

        This method contains an embedded Runge–Kutta scheme with orders 3 and
        4, which allows for an estimate of the local error.


        [1]: S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for step-size
        control in the interaction picture method," Computer Physics
        Communications, Volume 184, Issue 4, 2013, Pages 1211-1219

        https://doi.org/10.1016/j.cpc.2012.12.020
        """
        #--- Reference Phase and IP Operators
        ref_phase_beg = ref_phase
        ref_phase_mid = ref_phase_beg + self.dz_carrier_phase(z, 0.5*dz)
        ref_phase_end = ref_phase_mid + self.dz_carrier_phase(z+0.5*dz, 0.5*dz)

        IP_in_op_v = self.linear_operator(z, 0.5*dz)
        IP_out_op_v = self.linear_operator(z+dz, 0.5*dz)

        #--- Interaction Picture
        AI_v = IP_in_op_v * A_v

        #--- k1
        if k5_v is None:
            k5_v = self.nonlinear(A_v, z, ref_phase_beg)
        kI1_v = IP_in_op_v * k5_v

        #--- k2
        AI2_v = AI_v + (0.5*dz)*kI1_v
        kI2_v = self.nonlinear(AI2_v, z+0.5*dz, ref_phase_mid)

        #--- k3
        AI3_v = AI_v + (0.5*dz)*kI2_v
        kI3_v = self.nonlinear(AI3_v, z+0.5*dz, ref_phase_mid)

        #--- k4
        AI4_v = AI_v + dz*kI3_v
        A4_v = IP_out_op_v * AI4_v
        k4_v = self.nonlinear(A4_v, z+dz, ref_phase_end)

        #--- RK4
        bI_v = AI_v + (dz/6)*(kI1_v + 2*(kI2_v + kI3_v))
        b_v = IP_out_op_v * bI_v
        A_RK4_v = b_v + dz*k4_v/6

        #--- k5
        k5_v = self.nonlinear(A_RK4_v, z+dz, ref_phase_end)

        #--- RK3
        A_RK3_v = b_v + (dz/30)*(2*k4_v + 3*k5_v)

        return A_RK4_v, A_RK3_v, k5_v, ref_phase_end


    def dz_carrier_phase(self, z, dz):
        '''Returns the phase accumulated on the envelope due to the velocity
        mismatch between the carrier phase velocity and the retarted frame
        velocity.
        '''
        beta0 = self.beta(z=z)[self.ref_v_idx]
        beta1 = self.beta1(z=z)
        phase = -dz*(beta0 - self.ref_w*beta1)
        return phase


    def linear_operator(self, z, dz):
        '''Returns the linear operator in the frequency domain.'''
        # Loss
        a = self.alpha(z=z)

        # Phase
        beta = self.beta(z=z)
        beta0 = beta[self.ref_v_idx]
        beta1 = self.beta1(z=z)
        b = beta - (beta0 + beta1*self.af_grid)

        l_v = (a/2 - 1j*b) * dz
        exp_l_v = exp(l_v)
        return exp_l_v


    def nonlinear(self, A_v_analytic, z, ref_phase):
        '''Returns the result of the nonlinear operator in the frequency
        domain.
        '''
        #--- Electic Field
        #-----------------
        ref_exp_phase = exp(1j * ref_phase)
        A_v = (0.5 * ref_exp_phase) * A_v_analytic

        nl_E_v = np.zeros(self.nl_n_points, dtype=np.float)
        nl_E_v[self.slice_real] = A_v.real
        nl_E_v[self.slice_imag] = A_v.imag
        nl_E_t = mkl_fft.irfft(
            nl_E_v * (self.nl_df*self.nl_n_points),
            overwrite_x=True)

        #--- Chi2
        #--------
        if self.gamma2 is not None:
            if self.dispersive_chi2:
                nl2_ER2_v = nl_E_v * self.R2_v
                nl2_ER2_t = mkl_fft.irfft(
                    nl2_ER2_v * (self.nl_df*self.nl_n_points),
                    overwrite_x=True)
            else:
                nl2_ER2_t = nl_E_t

            nl2_E2R2_t = nl_E_t * nl2_ER2_t
            nl2_E2R2_v = mkl_fft.rfft(
                nl2_E2R2_t * self.nl_dt,
                overwrite_x=True)

            E2R2_v = np.empty(self.n_points, dtype=np.complex)
            E2R2_v.real = nl2_E2R2_v[self.slice_real]
            E2R2_v.imag = nl2_E2R2_v[self.slice_imag]
            E2R2_v *= 2

            g2_v = self.gamma2(z=z)

            nl2_v = E2R2_v * g2_v

        #--- Chi3
        #--------
        if self.gamma3 is not None:
            nl3_E2_t = nl_E_t**2
            if self.dispersive_chi3:
                nl3_E2_v = mkl_fft.rfft(
                    nl3_E2_t * self.nl_dt,
                    overwrite_x=True)
                nl3_E2R3_v = nl3_E2_v * self.R3_v #TODO: filter high frequency?
                nl3_E2R3_t = mkl_fft.irfft(
                    nl3_E2R3_v * (self.nl_df*self.nl_n_points),
                    overwrite_x=True)
            else:
                nl3_E2R3_t = nl3_E2_t

            nl3_E3R3_t = nl_E_t * nl3_E2R3_t
            nl3_E3R3_v = mkl_fft.rfft(
                nl3_E3R3_t * self.nl_dt,
                overwrite_x=True)
            # Return to analytic format
            AR3_v = np.empty(self.n_points, dtype=np.complex)
            AR3_v.real = nl3_E3R3_v[self.slice_real]
            AR3_v.imag = nl3_E3R3_v[self.slice_imag]
            AR3_v *= 2

            g3_v = self.gamma3(z=z)

            nl3_v = AR3_v * g3_v

        #--- Nonlinear Operator
        #----------------------
        if (self.gamma2 is not None) and (self.gamma3 is not None):
            nl_v = np.conj(ref_exp_phase) * -1j*(nl2_v + nl3_v)
        elif (self.gamma2 is not None):
            nl_v = np.conj(ref_exp_phase) * -1j*nl2_v
        elif (self.gamma3 is not None):
            nl_v = np.conj(ref_exp_phase) * -1j*nl3_v
        else:
            nl_v = 0j
        return nl_v

    def nonlinear_analytic(self, A_v, z, ref_phase):
        '''Returns the result of the nonlinear operator in the FFT frequency
        domain.

        The Chi2 interaction needs atleast 2x sampling to eliminate aliasing,
        and the Chi3 needs atleast 3x.
        '''
        ref_exp_phase = exp(1j * ref_phase)
        #--- Chi2
        if self.gamma2 is not None:
            nl2_A_v = resample_spectrum(ifftshift(A_v), self.nl_n_points)
            nl2_A_t = mkl_fft.ifft(nl2_A_v * self.nl_df*self.nl_n_points, overwrite_x=True)
            nl2_ref_exp_phase = self.nl_ref_exp_phase * ref_exp_phase

            nl2_AER2_t = nl2_A_t*(nl2_A_t.conj()*nl2_ref_exp_phase.conj() + 0.5*nl2_A_t*nl2_ref_exp_phase)
#            nl2_AER2_t *= self.nl_blackman
            nl2_AER2_v = mkl_fft.fft(nl2_AER2_t * self.nl_dt, overwrite_x=True)

            AER2_v = resample_spectrum(nl2_AER2_v, self.n_points)

            g2_v = self.gamma2(z=z)

            nl2_v = g2_v * fftshift(AER2_v)
        else:
            nl2_v = 0j

        #--- Chi3
        if self.gamma3 is not None:
            nl3_A_v = resample_spectrum(ifftshift(A_v), self.nl_n_points)
            nl3_A_t = mkl_fft.ifft(nl3_A_v * self.nl_df*self.nl_n_points, overwrite_x=True)

            nl3_ref_exp_phase = self.nl_ref_exp_phase * ref_exp_phase

            nl3_AE2R3_t = nl3_A_t*(0.25*(nl3_A_t*nl3_ref_exp_phase)**2 + 0.75*nl3_A_t*nl3_A_t.conj() + 0.75*(nl3_A_t*nl3_ref_exp_phase).conj()**2)
#            nl3_AE2R3_t *= self.nl_blackman
            nl3_AE2R3_v = mkl_fft.fft(nl3_AE2R3_t * self.nl_dt, overwrite_x=True)

            AE2R3_v = resample_spectrum(nl3_AE2R3_v, self.n_points)

            g3_v = self.gamma3(z=z)

            nl3_v = g3_v * fftshift(AE2R3_v)
        else:
            nl3_v = 0j

        #--- Nonlinear Operator
        nl_v = 1j*(nl2_v + nl3_v)
        return nl_v


    def calculate_nonlinear_susceptibility(self):
        '''Returns the RFFT of the normalized nonlinear susceptibilities chi2
        and chi3 in the frequency domain.

        The returned values are normalized such that sum(R*dt) = 1. Common
        functions are the instantaneous (dirac delta) and raman responses.

        Notes
        -----
        These relations only contain the nonlinear dispersion of the bulk
        material responses. Nonlinear dispersion attributable to the waveguide
        mode should be included in the gamma2 and gamma3 parameters.

        See `scipy.fftpack.rfft` for the strange storage implementation of the
        rfft.
        '''
        def dirac_delta(t_grid):
            dt = np.diff(t_grid).mean()
            dd_t = np.zeros_like(t_grid)
            dd_t[0] = 1
            dd_t *= 1/np.sum(dd_t * dt)
            return dd_t

        def R_a(t_grid, tau1, tau2, fraction):
            t_delay = t_grid - t_grid.min()
            RT = ((tau1**2+tau2**2)/(tau1*tau2**2))*exp(-t_delay/tau2)*np.sin(t_delay/tau1);
            RT *= fraction
            return RT

        #--- chi2
        # Instantaneous
        R2_v = mkl_fft.rfft(dirac_delta(self.nl_t_grid) * self.nl_dt, overwrite_x=True)

        #--- chi3
        # Instantaneous
        R3_v = mkl_fft.rfft(dirac_delta(self.nl_t_grid) * self.nl_dt, overwrite_x=True)

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
            raman_t += R_a(self.nl_t_grid, *weight)
        raman_t *= 1/np.sum(raman_t * self.nl_dt)
        raman_v = mkl_fft.rfft(raman_t * self.nl_dt, overwrite_x=True)
        R3_v = (1.-raman_fraction)*R3_v + raman_fraction*raman_v
        return R2_v, R3_v


    def pulse_energy(self):
        return np.sum(0.5 * np.abs(self.A_v)**2 * self.df)


    def estimate_step_size(self, A_v, z, local_error=1e-3, dz=1e-5):
        '''Estimate the optimal step size after integrating by a test dz'''
        #--- Integrate by dz
        A_RK4_v, A_RK3_v, k5_next_v, ref_phase_end = self.integrate(
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
                1e-12*(self.ref_v+np.gradient(np.unwrap(np.angle(self.A_t))/(2*pi), self.dt)),
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
                1e12*(self.t_window/2 - (np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.df) % self.t_window)),
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
                1e12*(self.t_window/2 - (np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.df) % self.t_window)),
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
                1e-12*(self.ref_v + np.gradient(np.unwrap(np.angle(self.A_t))/(2*pi), self.dt)))

        if self.plot_frq:
            self._ln_pwr.set_data(
                1e-12*self.v_grid,
                self.A_v.real**2+self.A_v.imag**2)
            self._ln_phs.set_data(
                1e-12*self.v_grid,
                1e12*(self.t_window/2 - (np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.df) % self.t_window)))

        if self.plot_wvl:
            self._ln_pwr.set_data(
                1e9*self.wl_grid,
                self.dJdHz_to_dJdm * (self.A_v.real**2+self.A_v.imag**2))
            self._ln_phs.set_data(
                1e9*self.wl_grid,
                1e12*(self.t_window/2 - (np.gradient(np.unwrap(np.angle(self.A_v))/(2*pi), self.df) % self.t_window)))

        # Update Z Label
        self._z_label.set_title('z = {:.9g} m'.format(self.z))

        # Blit
        for artist in self._artists:
            artist.axes.draw_artist(artist)

        self._rt_fig.canvas.blit(self._ax_0.bbox)
        self._rt_fig.canvas.blit(self._ax_1.bbox)
        self._rt_fig.canvas.start_event_loop(1e-6)

    def spectrogram(self, A_v_analytic, fwhm=100e-15, delays=None):
        # Analytic to Real
        A_v = 0.5*A_v_analytic

        E_v = np.zeros(self.cr_n_points, dtype=np.float)
        E_v[self.slice_real] = A_v.real
        E_v[self.slice_imag] = A_v.imag
        E_t = mkl_fft.irfft(
            E_v * (self.cr_df*self.cr_n_points),
            overwrite_x=True)

        # Delays
        if delays is None:
            d_min, d_max = self.t_grid.min(), self.t_grid.max()
            n_delays = int(10*round((d_max - d_min)/fwhm))
        elif len(delays)==2:
            d_min, d_max = (self.t_grid.min(), self.t_grid.max()) if (delays[0] is None) else delays[0]
            n_delays = int(10*round((d_max - d_min)/fwhm)) if (delays[1] is None) else delays[1]
        delays = np.linspace(d_min, d_max, n_delays)

        # Spectrogram
        window = gaussian_window(self.cr_t_grid[:, np.newaxis], delays[np.newaxis, :], fwhm)
#        return window
        print(window.shape)

        spec_E_v = mkl_fft.rfft(E_t[:, np.newaxis]*window * self.cr_dt, overwrite_x=True, axis=0)

        # Return to analytic format
        spec_A_v = np.empty((self.n_points, len(delays)), dtype=np.complex)
        spec_A_v.real = spec_E_v[self.slice_real, :]
        spec_A_v.imag = spec_E_v[self.slice_imag, :]
        spec_A_v *= 2

        # Power
        spec = spec_A_v.real**2 + spec_A_v.imag**2
        return spec

