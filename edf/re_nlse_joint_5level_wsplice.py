"""
let's just solve the forward pumped case first. the backward pumped case is an
iteration from using the forward pumped case anyways, as far as i understand
it anyways.
"""

# %% ----- imports
import numpy as np
from scipy.constants import c, h
import pynlo
from scipy.integrate import RK45
import collections
from edf.five_level_ss_eqns import (
    _n1_func,
    _n2_func,
    _n3_func,
    _n4_func,
    _n5_func,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
    xi_p,
    eps_p,
    eps_s,
)


ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

SimulationResult = collections.namedtuple(
    "SimulationResult",
    [
        "pulse",
        "z",
        "a_t",
        "a_v",
        "Pp",
        "n1_n",
        "n2_n",
        "n3_n",
        "n4_n",
        "n5_n",
        "g_v",
    ],
)


def package_sim_output(simulate):
    def wrapper(self, *args, **kwargs):
        (pulse_out, z, a_t, a_v, Pp, n1_n, n2_n, n3_n, n4_n, n5_n, g_v) = simulate(
            self, *args, **kwargs
        )
        model = self

        class result:
            def __init__(self):
                self.pulse_out = pulse_out.copy()
                self.z = z
                self.a_t = a_t
                self.a_v = a_v
                self.p_t = abs(a_t) ** 2
                self.p_v = abs(a_v) ** 2
                self.phi_v = np.angle(a_v)
                self.phi_t = np.angle(a_t)
                self.Pp = Pp
                self.n1_n = n1_n
                self.n2_n = n2_n
                self.n3_n = n3_n
                self.n4_n = n4_n
                self.n5_n = n5_n
                self.model = model
                self.g_v = g_v

            def animate(self, plot, save=False, p_ref=None):
                pynlo.utility.misc.animate(
                    self.pulse_out,
                    self.model,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    save=save,
                    p_ref=p_ref,
                )

            def plot(self, plot, num="Simulation Results"):
                return pynlo.utility.misc.plot_results(
                    self.pulse_out,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    num=num,
                )

            def save(self, path, filename):
                assert path != "" and isinstance(path, str), "give a save path"
                assert filename != "" and isinstance(filename, str)

                path = path + "/" if path[-1] != "" else path
                np.save(path + filename + "_t_grid.npy", self.pulse_out.t_grid)
                np.save(path + filename + "_v_grid.npy", self.pulse_out.v_grid)
                np.save(path + filename + "_z.npy", self.z)
                np.save(path + filename + "_amp_t.npy", abs(self.pulse_out.a_t))
                np.save(path + filename + "_amp_v.npy", abs(self.pulse_out.a_v))
                np.save(path + filename + "_phi_t.npy", np.angle(self.pulse_out.a_t))
                np.save(path + filename + "_phi_v.npy", np.angle(self.pulse_out.a_v))

        return result()

    return wrapper


class Mode(pynlo.media.Mode):
    def __init__(
        self,
        v_grid,
        beta,
        g2=None,
        g2_inv=None,
        g3=None,
        rv_grid=None,
        r3=None,
        z=0.0,
        # --------- parameters for EDF --------
        p_v=None,
        f_r=100e6,
        overlap_p=1.0,
        overlap_s=1.0,
        # --------- accounting for splicing of two doped fibers -----
        n_ion_1=7e24,
        n_ion_2=7e24,
        z_spl=0.0,
        loss_spl=0.0,
        a_eff_1=3.14e-12,
        a_eff_2=3.14e-12,
        # -----------------------------------------------------------
        sigma_p=None,
        sigma_a=None,
        sigma_e=None,
        Pp_fwd=0.0,
        eps_p=eps_p,
        xi_p=xi_p,
        eps_s=eps_s,
        tau_21=tau_21,
        tau_32=tau_32,
        tau_43=tau_43,
        tau_54=tau_54,
        sum_a_prev=None,
        sum_e_prev=None,
        Pp_prev=None,
    ):
        assert isinstance(p_v, (np.ndarray, pynlo.utility.misc.ArrayWrapper))
        assert p_v.size == v_grid.size
        assert sigma_e is not None, "input absorption cross-section at 980 nm"
        assert isinstance(sigma_a, np.ndarray), "provide absorption cross-section"
        assert isinstance(sigma_e, np.ndarray), "provide emission cross-section"
        assert (
            sigma_a.size == v_grid.size
        ), "absorption cross section grid must match frequency grid"
        assert (
            sigma_e.size == v_grid.size
        ), "emission cross section grid must match frequency grid"

        self.f_r = f_r
        self.overlap_p = overlap_p
        self.overlap_s = overlap_s
        self._n_ion_1 = n_ion_1
        self._n_ion_2 = n_ion_2
        self.z_spl = z_spl
        self.loss_spl = loss_spl  # to be applied in model when z=z_spl
        self._a_eff_1 = a_eff_1
        self._a_eff_2 = a_eff_2
        self.sigma_p = sigma_p
        self.sigma_a = sigma_a
        self.sigma_e = sigma_e
        self._Pp_fwd = Pp_fwd
        self._z_start = z
        self._rk45_Pp = None

        self._eps_p = eps_p
        self._xi_p = xi_p
        self._eps_s = eps_s
        self._tau_21 = tau_21
        self._tau_32 = tau_32
        self._tau_43 = tau_43
        self._tau_54 = tau_54

        if sum_a_prev is None:
            assert sum_e_prev is None, "cannot only supply one"
            sum_a_was_None = True
            sum_a_prev = lambda z: 0
        if sum_e_prev is None:
            assert sum_a_was_None, "cannot only supply one"
            sum_e_prev = lambda z: 0
        if Pp_prev is None:
            Pp_prev = lambda z: 0
        assert callable(sum_a_prev)
        assert callable(sum_e_prev)
        assert callable(Pp_prev)
        self.sum_a_prev = sum_a_prev
        self.sum_e_prev = sum_e_prev
        self.Pp_prev = Pp_prev

        # alpha = lambda z, p_v: self.gain
        super().__init__(v_grid, beta, self.gain, g2, g2_inv, g3, rv_grid, r3, z)
        # self.v_grid is not defined until after the __init__ call
        # __init__ sets _p_v to None, so assign this after the __init__ call
        self.p_v = p_v
        self.dv = self.v_grid[1] - self.v_grid[0]

    @property
    def n_ion(self):
        if self.z < self.z_spl:
            return self._n_ion_1
        else:
            return self._n_ion_2

    @property
    def a_eff(self):
        if self.z < self.z_spl:
            return self._a_eff_1
        else:
            return self._a_eff_2

    @property
    def tau_21(self):
        return self._tau_21

    @tau_21.setter
    def tau_21(self, tau):
        self._tau_21 = tau

    @property
    def tau_32(self):
        return self._tau_32

    @tau_32.setter
    def tau_32(self, tau):
        self._tau_32 = tau

    @property
    def tau_43(self):
        return self._tau_43

    @tau_43.setter
    def tau_43(self, tau):
        self._tau_43 = tau

    @property
    def tau_54(self):
        return self._tau_54

    @tau_54.setter
    def tau_54(self, tau):
        self._tau_54 = tau

    @property
    def eps_p(self):
        return self._eps_p

    @eps_p.setter
    def eps_p(self, eps_p):
        self._eps_p = eps_p

    @property
    def xi_p(self):
        return self._xi_p

    @xi_p.setter
    def xi_p(self, xi_p):
        self._xi_p = xi_p

    @property
    def eps_s(self):
        return self._eps_s

    @eps_s.setter
    def eps_s(self, eps_s):
        self._eps_s = eps_s

    @property
    def nu_p(self):
        return c / 980e-9

    @property
    def _sum_a(self):
        p_s = self.f_r * self.p_v * self.dv
        sum_a = self.overlap_s * p_s * self.sigma_a / (h * self.v_grid * self.a_eff)
        sum_a = np.sum(sum_a)
        return sum_a

    @property
    def _sum_e(self):
        p_s = self.f_r * self.p_v * self.dv
        sum_e = self.overlap_s * p_s * self.sigma_e / (h * self.v_grid * self.a_eff)
        sum_e = np.sum(sum_e)
        return sum_e

    @property
    def n1(self):
        sum_a = self._sum_a + self.sum_a_prev(self.z)
        sum_e = self._sum_e + self.sum_e_prev(self.z)
        n1 = _n1_func(
            self.n_ion,
            self.a_eff,
            self.overlap_p,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.eps_p,
            self.xi_p,
            self.eps_s,
            self.tau_21,
            self.tau_32,
            self.tau_43,
            self.tau_54,
        )
        return n1

    @property
    def n2(self):
        sum_a = self._sum_a + self.sum_a_prev(self.z)
        sum_e = self._sum_e + self.sum_e_prev(self.z)
        n2 = _n2_func(
            self.n_ion,
            self.a_eff,
            self.overlap_p,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.eps_p,
            self.xi_p,
            self.eps_s,
            self.tau_21,
            self.tau_32,
            self.tau_43,
            self.tau_54,
        )
        return n2

    @property
    def n3(self):
        sum_a = self._sum_a + self.sum_a_prev(self.z)
        sum_e = self._sum_e + self.sum_e_prev(self.z)
        n3 = _n3_func(
            self.n_ion,
            self.a_eff,
            self.overlap_p,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.eps_p,
            self.xi_p,
            self.eps_s,
            self.tau_21,
            self.tau_32,
            self.tau_43,
            self.tau_54,
        )
        return n3

    @property
    def n4(self):
        sum_a = self._sum_a + self.sum_a_prev(self.z)
        sum_e = self._sum_e + self.sum_e_prev(self.z)
        n4 = _n4_func(
            self.n_ion,
            self.a_eff,
            self.overlap_p,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.eps_p,
            self.xi_p,
            self.eps_s,
            self.tau_21,
            self.tau_32,
            self.tau_43,
            self.tau_54,
        )
        return n4

    @property
    def n5(self):
        sum_a = self._sum_a + self.sum_a_prev(self.z)
        sum_e = self._sum_e + self.sum_e_prev(self.z)
        n5 = _n5_func(
            self.n_ion,
            self.a_eff,
            self.overlap_p,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.eps_p,
            self.xi_p,
            self.eps_s,
            self.tau_21,
            self.tau_32,
            self.tau_43,
            self.tau_54,
        )
        return n5

    def gain(self, z, p_v):
        return (
            -self.sigma_a * self.n1
            + self.sigma_e * self.n2
            - self.sigma_a * self.eps_s * self.n2
        ) * self.overlap_s

    def _dPp_dz(self, z, Pp):
        deriv = (
            (
                -self.sigma_p * self.n1
                + self.sigma_p * self.xi_p * self.n3
                - self.sigma_p * self.eps_p * self.n3
            )
            * self.overlap_p
            * Pp
        )
        return deriv

    def setup_rk45_Pp(self, dz):
        self._rk45_Pp = RK45(
            fun=self._dPp_dz,
            t0=self._z_start,
            y0=np.array([self.Pp_fwd]),
            t_bound=np.inf,
            max_step=dz,
        )

    @property
    def rk45_Pp(self):
        assert self._rk45_Pp is not None, "setup rk45 by calling setup_rk45_Pp(dz)"
        return self._rk45_Pp

    @property
    def Pp(self):
        return self.Pp_fwd + self.Pp_prev(self.z)

    @property
    def Pp_fwd(self):
        if self._rk45_Pp is not None:
            self._Pp_fwd = self.rk45_Pp.y[0]
        return self._Pp_fwd

    def update_Pp(self):
        while self.rk45_Pp.t < self.z:
            self.rk45_Pp.step()


class Model_EDF(pynlo.model.Model):
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)
        self._Pp_record = []
        self._sum_a_record = []
        self._sum_e_record = []
        self._z_record = []
        self.loss_spl_applied = False

    @property
    def Pp_record(self):
        return np.asarray(self._Pp_record)

    @property
    def sum_a_record(self):
        return np.asarray(self._sum_a_record)

    @property
    def sum_e_record(self):
        return np.asarray(self._sum_e_record)

    @property
    def z_record(self):
        return np.asarray(self._z_record)

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        """
        Propagate the given pulse spectrum from `z` to `z_stop` using an
        adaptive step size algorithm.

        The step size algorithm utilizes an embedded Runge–Kutta scheme with
        orders 3 and 4 (ERK4(3)-IP) [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The starting point.
        z_stop : float
            The stopping point.
        dz : float
            The initial step size.
        local_error : float
            The relative local error of the adaptive step size algorithm.
        k5_v : ndarray of complex, optional
            The action of the nonlinear operator on the solution from the
            preceding step. The default is ``None``.
        cont : bool, optional
            A flag that indicates the current step is continuous with the
            previous, i.e. that it begins where the other ended. The default is
            ``False``.

        Returns
        -------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The z position in the mode.
        dz : float
            The step size.
        k5_v : ndarray of complex
            The nonlinear action of the 4th-order result.
        cont : bool
            A flag indicating that the next step may be continuous.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219
            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        p_v = abs(a_v) ** 2
        if self._use_fftshift:
            p_v = np.fft.fftshift(p_v)
        self.mode.p_v[:] = p_v[:]
        self.mode.update_Pp()

        while z < z_stop:
            # Don't let the simulation step by more than 1 mm! This is to help
            # force it sync up with the pump's rk45. The other option is to
            # encode the pump update into update_linearity() which is called
            # during step(). However, this is not so easy to do because the
            # pump is not just a callable function, but a value calculated
            # using it's own rk45.
            dz = min([dz, 1e-3])

            z_next = z + dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz_adaptive = dz  # save value of last step size
                dz = z_next - z  # force smaller step size to hit z_stop
            else:
                final_step = False

            # ---- Integrate by dz
            a_RK4_v, a_RK3_v, k5_v_next = self.step(
                a_v, z, z_next, k5_v=k5_v, cont=cont
            )

            # ---- Estimate Relative Local Error
            est_error = pynlo.model.l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error / local_error) ** 0.25

            # ---- Propagate Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                dz = dz / 2
                cont = False
            else:
                # Update parameters for the next loop
                z = z_next
                a_v = a_RK4_v
                k5_v = k5_v_next
                if (not final_step) or (error_ratio > 1):
                    dz = dz / max(error_ratio, 0.5)
                else:
                    dz = dz_adaptive  # if final step, use adaptive step size
                cont = True

                # ----------- if this loop passed, update values needed to
                #             calculate the next one!

                # update pulse energy for gain calculation
                p_v[:] = abs(a_v) ** 2
                if self._use_fftshift:
                    p_v = np.fft.fftshift(p_v)
                self.mode.p_v[:] = p_v[:]
                self.mode.update_Pp()

                # apply loss if z > z_spl
                if z > self.mode.z_spl:
                    if not self.loss_spl_applied:
                        a_v *= self.mode.loss_spl**0.5
                        self.mode.rk45_Pp.y *= self.mode.loss_spl
                        self.loss_spl_applied = True

                # record values for future sims
                self._sum_a_record.append(self.mode._sum_a)
                self._sum_e_record.append(self.mode._sum_e)
                self._Pp_record.append(self.mode.Pp_fwd)
                self._z_record.append(z)

        return a_v, z, dz, k5_v, cont

    @package_sim_output
    def simulate(self, z_grid, local_error=1e-6, n_records=None, plot=None):
        """
        Simulate propagation of the input pulse through the optical mode.

        Parameters
        ----------
        z_grid : float or array_like of floats
            The total propagation distance over which to simulate, or the z
            positions at which to solve for the pulse spectrum. An adaptive
            step-size algorithm is used to propagate between these points. If
            only the end point is given the starting point is assumed to be the
            origin.
        local_error : float, optional
            The target relative local error for the adaptive step size
            algorithm. The default is 1e-6.
        n_records : None or int, optional
            The number of simulation points to return. If set, the z positions
            will be linearly spaced between the first and last points of
            `z_grid`. If ``None``, the default is to return all points as
            defined in `z_grid`. The record always includes the starting and
            ending points.
        plot : None or string, optional
            A flag that activates real-time visualization of the simulation.
            The options are ``"frq"``, ``"time"``, or ``"wvl"``, corresponding
            to the frequency, time, and wavelength domains. If set, the plot is
            updated each time the simulation reaches one of the z positions
            returned at the output. If ``None``, the default is to run the
            simulation without real-time plotting.

        Returns
        -------
        pulse : :py:class:`~pynlo.light.Pulse`
            The output pulse. This object can be used as the input to another
            simulation.
        z : ndarray of float
            The z positions at which the pulse spectrum (`a_v`) and complex
            envelope (`a_t`) have been returned.
        a_t : ndarray of complex
            The root-power complex envelope of the pulse at each z position.
        a_v : ndarray of complex
            The root-power spectrum of the pulse at each z position.
        """
        # ---- Z Grid
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.size == 1:
            # Since only the end point was given, the start point is the origin
            z_grid = np.append(0.0, z_grid)

        if n_records is None:
            n_records = z_grid.size
            z_record = z_grid
        else:
            assert n_records >= 2, "The output must include atleast 2 points."
            z_record = np.linspace(z_grid.min(), z_grid.max(), n_records)
            z_grid = np.unique(np.append(z_grid, z_record))
        z_record = {z: idx for idx, z in enumerate(z_record)}

        if self.mode.z_nonlinear.pol:  # support subclasses with poling
            # always simulate up to the edge of a poled domain
            z_grid = np.unique(np.append(z_grid, list(self.mode.g2_inv)))

        # splice point needs to be within the length of the fiber!
        # assert self.mode.z_spl < z_grid[-1], "splice point needs to be in the fiber"

        # ---- Setup
        z = z_grid[0]
        pulse_out = self.pulse.copy()

        # Frequency Domain
        a_v_record = np.empty((n_records, pulse_out.n), dtype=complex)
        a_v_record[0, :] = pulse_out.a_v

        # Time Domain
        a_t_record = np.empty((n_records, pulse_out.n), dtype=complex)
        a_t_record[0, :] = pulse_out.a_t

        # Pump power
        Pp = np.empty(n_records, dtype=float)
        Pp[0] = self.mode.Pp_fwd

        # inversion
        n1_n = np.empty(n_records, dtype=float)
        n1_n[0] = self.mode.n1 / self.mode.n_ion
        n2_n = np.empty(n_records, dtype=float)
        n2_n[0] = self.mode.n2 / self.mode.n_ion
        n3_n = np.empty(n_records, dtype=float)
        n3_n[0] = self.mode.n3 / self.mode.n_ion
        n4_n = np.empty(n_records, dtype=float)
        n4_n[0] = self.mode.n4 / self.mode.n_ion
        n5_n = np.empty(n_records, dtype=float)
        n5_n[0] = self.mode.n5 / self.mode.n_ion

        # gain
        g_v = np.empty((n_records, pulse_out.n), dtype=float)
        g_v[0, :] = self.mode.gain(None, None)

        # Step Size
        dz = 1e-3

        # Plotting
        if plot is not None:
            assert plot in ["frq", "time", "wvl"], (
                "Plot choice '{:}' is unrecognized"
            ).format(plot)
            # Setup Plots
            self._setup_plots(plot, pulse_out, z)

        # ---- Propagate
        k5_v = None
        cont = False
        for z_stop in z_grid[1:]:
            # Step
            (pulse_out.a_v, z, dz, k5_v, cont) = self.propagate(
                pulse_out.a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont
            )

            # Record
            if z in z_record:
                idx = z_record[z]
                a_t_record[idx, :] = pulse_out.a_t
                a_v_record[idx, :] = pulse_out.a_v
                Pp[idx] = self.mode.Pp_fwd
                n1_n[idx] = self.mode.n1 / self.mode.n_ion
                n2_n[idx] = self.mode.n2 / self.mode.n_ion
                n3_n[idx] = self.mode.n3 / self.mode.n_ion
                n4_n[idx] = self.mode.n4 / self.mode.n_ion
                n5_n[idx] = self.mode.n5 / self.mode.n_ion
                g_v[idx, :] = self.mode.gain(None, None)

                # Plot
                if plot is not None:
                    # Update Plots
                    self._update_plots(plot, pulse_out, z)

                    if z == z_grid[-1]:
                        # End animation with the last step
                        for artist in self._artists:
                            artist.set_animated(False)

        sim_res = SimulationResult(
            pulse=pulse_out,
            z=np.fromiter(z_record.keys(), dtype=float),
            a_t=a_t_record,
            a_v=a_v_record,
            Pp=Pp,
            n1_n=n1_n,
            n2_n=n2_n,
            n3_n=n3_n,
            n4_n=n4_n,
            n5_n=n5_n,
            g_v=g_v,
        )
        return sim_res


class NLSE(pynlo.model.NLSE):
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)
        self._Pp_record = []
        self._sum_a_record = []
        self._sum_e_record = []
        self._z_record = []
        self.loss_spl_applied = False

    @property
    def Pp_record(self):
        return np.asarray(self._Pp_record)

    @property
    def sum_a_record(self):
        return np.asarray(self._sum_a_record)

    @property
    def sum_e_record(self):
        return np.asarray(self._sum_e_record)

    @property
    def z_record(self):
        return np.asarray(self._z_record)

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        # ---- Standard FFT Order
        a_v = np.fft.ifftshift(a_v)
        self._use_fftshift = True

        # ---- Propagate
        a_v, z, dz, k5_v, cont = Model_EDF.propagate(
            self, a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont
        )

        # ---- Monotonic Order
        a_v = np.fft.fftshift(a_v)
        return a_v, z, dz, k5_v, cont

    def simulate(self, z_grid, local_error=1e-6, n_records=None, plot=None):
        return Model_EDF.simulate(self, z_grid, local_error, n_records, plot)


class EDF(pynlo.materials.SilicaFiber):
    def __init__(
        self,
        f_r=100e6,
        overlap_p=1.0,
        overlap_s=1.0,
        # --------- accounting for splicing of two doped fibers -----
        n_ion_1=7e24,
        n_ion_2=7e24,
        z_spl=0.0,
        loss_spl=0.0,
        a_eff_1=3.14e-12,
        a_eff_2=3.14e-12,
        gamma_1=0,
        gamma_2=0,
        # -----------------------------------------------------------
        sigma_p=None,
        sigma_a=None,
        sigma_e=None,
        eps_p=eps_p,
        xi_p=xi_p,
        eps_s=eps_s,
        tau_21=tau_21,
        tau_32=tau_32,
        tau_43=tau_43,
        tau_54=tau_54,
    ):
        super().__init__()

        self.f_r = f_r
        self.overlap_p = overlap_p
        self.overlap_s = overlap_s
        self._n_ion_1 = n_ion_1
        self._n_ion_2 = n_ion_2
        self.z_spl = z_spl
        self.loss_spl = loss_spl
        self._a_eff_1 = a_eff_1
        self._a_eff_2 = a_eff_2
        self._gamma_1 = gamma_1
        self._gamma_2 = gamma_2
        self.sigma_p = sigma_p
        self.sigma_a = sigma_a
        self.sigma_e = sigma_e
        self._tau_21 = tau_21
        self._tau_32 = tau_32
        self._tau_43 = tau_43
        self._tau_54 = tau_54
        self._xi_p = xi_p
        self._eps_p = eps_p
        self._eps_s = eps_s

    @property
    def tau_21(self):
        return self._tau_21

    @tau_21.setter
    def tau_21(self, tau):
        self._tau_21 = tau

    @property
    def tau_32(self):
        return self._tau_32

    @tau_32.setter
    def tau_32(self, tau):
        self._tau_32 = tau

    @property
    def tau_43(self):
        return self._tau_43

    @tau_43.setter
    def tau_43(self, tau):
        self._tau_43 = tau

    @property
    def tau_54(self):
        return self._tau_54

    @tau_54.setter
    def tau_54(self, tau):
        self._tau_54 = tau

    @property
    def eps_p(self):
        return self._eps_p

    @eps_p.setter
    def eps_p(self, eps_p):
        self._eps_p = eps_p

    @property
    def xi_p(self):
        return self._xi_p

    @xi_p.setter
    def xi_p(self, xi_p):
        self._xi_p = xi_p

    @property
    def eps_s(self):
        return self._eps_s

    @eps_s.setter
    def eps_s(self, eps_s):
        self._eps_s = eps_s

    def g3(self, v_grid, t_shock=None):
        """
        g3 nonlinear parameter

        Args:
            v_grid (1D array):
                frequency grid
            t_shock (float, optional):
                the characteristic time scale of optical shock formation, default is None
                in which case it is taken to be 1 / (2 pi v0)

        Returns:
            g3
        """
        g3_1 = pynlo.utility.chi3.gamma_to_g3(v_grid, self._gamma_1, t_shock=t_shock)
        g3_2 = pynlo.utility.chi3.gamma_to_g3(v_grid, self._gamma_2, t_shock=t_shock)
        return lambda z: g3_1 if z < self.z_spl else g3_2

    def generate_model(
        self,
        pulse,
        beta_1,
        beta_2,
        t_shock="auto",
        raman_on=True,
        Pp_fwd=0,
        sum_a_prev=None,
        sum_e_prev=None,
        Pp_prev=None,
    ):
        """
        generate pynlo.model.UPE or NLSE instance

        Args:
            pulse (object):
                instance of pynlo.light.Pulse
            beta_1 (np.ndarray):
                beta for first fiber, must be array that matches pulse's v_grid
            beta_2 (np.ndarray):
                beta for second fiber, must be array that matches pulse's v_grid
            t_shock (float, optional):
                time for optical shock formation, defaults to 1 / (2 pi pulse.v0)
            raman_on (bool, optional):
                whether to include raman effects, default is True
            alpha (array or callable, optional):
                default is 0, otherwise is a callable alpha(z, e_p) that returns a
                float or array, or fixed alpha.
        Returns:
            model
        """
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse

        if isinstance(t_shock, str):
            assert t_shock.lower() == "auto"
            t_shock = 1 / (2 * np.pi * pulse.v0)
        else:
            assert isinstance(t_shock, float) or t_shock is None

        analytic = True
        n = pulse.n
        dt = pulse.dt

        v_grid = pulse.v_grid
        assert isinstance(beta_1, np.ndarray) and beta_1.size == pulse.n
        assert isinstance(beta_2, np.ndarray) and beta_2.size == pulse.n
        beta = lambda z: beta_1 if z < self.z_spl else beta_2
        # if beta is None:
        #     beta = self.beta(v_grid)
        g3 = self.g3(v_grid, t_shock=t_shock)
        if raman_on:
            rv_grid, raman = self.raman(n, dt, analytic=analytic)
        else:
            rv_grid = raman = None

        mode = Mode(
            v_grid,
            beta,
            g2=None,
            g2_inv=None,
            g3=g3,
            rv_grid=rv_grid,
            r3=raman,
            z=0.0,
            # --------- parameters for EDF --------
            p_v=pulse.p_v.copy(),
            f_r=self.f_r,
            overlap_p=self.overlap_p,
            overlap_s=self.overlap_s,
            n_ion_1=self._n_ion_1,
            n_ion_2=self._n_ion_2,
            z_spl=self.z_spl,
            loss_spl=self.loss_spl,
            a_eff_1=self._a_eff_1,
            a_eff_2=self._a_eff_2,
            sigma_p=self.sigma_p,
            sigma_a=self.sigma_a,
            sigma_e=self.sigma_e,
            Pp_fwd=Pp_fwd,
            eps_p=self.eps_p,
            xi_p=self.xi_p,
            eps_s=self.eps_s,
            tau_21=self.tau_21,
            tau_32=self.tau_32,
            tau_43=self.tau_43,
            tau_54=self.tau_54,
            sum_a_prev=sum_a_prev,
            sum_e_prev=sum_e_prev,
            Pp_prev=Pp_prev,
        )

        # print("USING NLSE")
        model = NLSE(pulse, mode)
        model.mode.setup_rk45_Pp(1e-3)
        return model
