from scipy.special import erf
import numpy as np
import scipy.constants as sc
import pynlo


def LN_alpha(v_grid):
    """
    This is the absorption coefficient of ppln. When doing DFG and such, you
    deal with pretty low powers in the MIR, so it actually becomes problematic
    if you let the simulation generate arbitrarily long wavelengths. I ended
    uphaving to include the absorption coefficient, which is defined here.

    Args:
        v_grid (1D array):
            frequency grid

    Returns:
        1D array:
            absorption coefficient
    """
    w_grid = v_grid * 2 * np.pi
    w_grid /= 1e12
    return 1e6 * (1 + erf(-(w_grid - 300.0) / (10 * np.sqrt(2))))


def gbeam_area_scaling(z_to_focus, v0, a_eff):
    """
    A gaussian beam can be accounted for by scaling the chi2 and chi3 parameter

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        returns 1 / (current area / original area)
    """
    w_0 = np.sqrt(a_eff / np.pi)  # beam radius
    wl = sc.c / v0
    z_R = np.pi * w_0**2 / wl  # rayleigh length
    w = w_0 * np.sqrt(1 + (z_to_focus / z_R) ** 2)
    return 1 / (np.pi * w**2 / a_eff)


def chi2_gbeam_scaling(z_to_focus, v0, a_eff):
    """
    scaling for the chi2 parameter for gaussian beam

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        The chi2 parameter scales as 1 / sqrt[a_eff]
    """
    return gbeam_area_scaling(z_to_focus, v0, a_eff) ** 0.5


def chi3_gbeam_scaling(z_to_focus, v0, a_eff):
    """
    scaling for the chi3 parameter for gaussian beam

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        The chi3 parameter scales as 1 / a_eff. This is the same as chi2 but
        without the square root
    """
    return gbeam_area_scaling(z_to_focus, v0, a_eff)


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


def n_cLN(v_grid, T=24.5, axis="e"):
    """
    Refractive index of congruent lithium niobate.

    References
    ----------
    Dieter H. Jundt, "Temperature-dependent Sellmeier equation for the index of
     refraction, ne, in congruent lithium niobate," Opt. Lett. 22, 1553-1555
     (1997). https://doi.org/10.1364/OL.22.001553

    """

    assert axis == "e"

    a1 = 5.35583
    a2 = 0.100473
    a3 = 0.20692
    a4 = 100.0
    a5 = 11.34927
    a6 = 1.5334e-2
    b1 = 4.629e-7
    b2 = 3.862e-8
    b3 = -0.89e-8
    b4 = 2.657e-5

    wvl = sc.c / v_grid * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        a1
        + b1 * f
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


class MgLN:
    def __init__(self, T=24.5, axis="e"):
        self._T = T
        self._axis = axis

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        """
        set the temperature in Celsius

        Args:
            val (float):
                the temperature in Celsius
        """
        assert isinstance(val, float)
        self._T = val

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        """
        set the axis to be either extraordinary or ordinary

        Args:
            val (string):
                either "e" or "o"
        """
        assert np.any([val == "e", val == "o"]), 'the axis must be either "e" or "o"'
        self._axis = val

    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis)

    @property
    def beta(self):
        """
        Returns:
            callable:
                a function that calculates the angular wavenumber as a function
                of frequency
        """
        # n * omega * c
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis) * 2 * np.pi * v / sc.c

    @property
    def d_eff(self):
        """
        d_eff of magnesium doped lithium niobate

        Returns:
            float: d_eff
        """
        return 27e-12  # 27 pm / V

    @property
    def chi2_eff(self):
        """
        effective chi2 of magnesium doped lithium niobate

        Returns:
            float: 2 * d_eff
        """
        return 2 * self.d_eff

    @property
    def chi3_eff(self):
        """
        3rd order nonlinearity of magnesium doped lithium niobate

        Returns:
            float
        """
        return 5200e-24  # 5200 pm ** 2 / V ** 2

    def g2_shg(self, v_grid, v0, a_eff):
        """
        The 2nd order nonlinear parameter weighted for second harmonic
        generation driven by the given input frequency.

        Args:
            v_grid (1D array):
                frequency grid
            v0 (float):
                center frequency
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        return pynlo.utility.chi2.g2_shg(
            v0, v_grid, self.n(v_grid), a_eff, self.chi2_eff
        )

    def g3(self, v_grid, a_eff):
        """
        The 3rd order nonlinear parameter weighted for self-phase modulation.

        Args:
            v_grid (1D array):
                frequency grid
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        n_eff = self.n(v_grid)
        return pynlo.utility.chi3.g3_spm(n_eff, a_eff, self.chi3_eff)

    def generate_model(
        self,
        pulse,
        a_eff,
        length,
        g2_inv=None,
        beta=None,
        is_gaussian_beam=False,
    ):
        """
        generate PyNLO model instance

        Args:
            pulse (object):
                PyNLO pulse instance
            a_eff (float):
                effective area
            length (float):
                crystal or waveguide length
            g2_inv (1D array, optional):
                locations of all inversion points inside the crystal, default
                is no inversion points
            beta (1D array, optional):
                the beta curve calculated over the pulse's frequency grid. The
                default is None, in which case beta is calculated from Mg-LN's
                material dispersion.
            is_gaussian_beam (bool, optional):
                whether the mode is a gaussian beam, default is False

        Returns:
            model (object):
                a PyNLO model instance

        Notes:
            polling_period and polling_sign_callable cannot both be provided,
            if "is_gaussian_beam" is set to True, then the chi2 parameter is
            scaled by the ratio of effective areas^1/2 as a function of z

            If is_gaussian_beam is not True, then it is assumed that
            propagation occurs inside a waveguide, in which case an assert
            statement checks that the beta curve was provided
        """
        # --------- assert statements ---------
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse

        # ------ g2 and g3---------
        g2_array = self.g2_shg(pulse.v_grid, pulse.v0, a_eff)
        g3_array = self.g3(pulse.v_grid, a_eff)

        # make g2 and g3 callable if the mode is a gaussian beam
        if is_gaussian_beam:

            def g2_func(z):
                z_to_focus = z - length / 2
                return g2_array * chi2_gbeam_scaling(z_to_focus, pulse.v0, a_eff)

            def g3_func(z):
                z_to_focus = z - length / 2
                return g3_array * chi3_gbeam_scaling(z_to_focus, pulse.v0, a_eff)

            g2 = g2_func
            g3 = g3_func

        else:
            g2 = g2_array
            g3 = g3_array

            if beta is None:
                msg = (
                    "IF NOT GAUSSIAN BEAM, WAVEGUIDE DISPERSION SHOULD BE"
                    + " ACCOUNTED FOR BY PROVIDING THE BETA CURVE, BUT NONE WAS PROVIDED"
                )
                print(msg)

        # ----- mode and model ---------
        if beta is None:
            # calculate beta from material dispersion
            beta = self.beta(pulse.v_grid)
        else:
            # beta is already provided
            assert isinstance(beta, np.ndarray) and beta.shape == pulse.v_grid.shape

        mode = pynlo.media.Mode(
            pulse.v_grid,
            beta,
            alpha=-LN_alpha(pulse.v_grid),
            g2=g2,  # callable if gaussian beam
            g2_inv=g2_inv,  # callable
            g3=g3,  # callable if gaussian beam
            z=0.0,
        )

        model = pynlo.model.UPE(pulse, mode)
        return model


class cLN(MgLN):
    def __init__(self, T=24.5, axis="e"):
        super().__init__(T=T, axis=axis)

    # overides n in MgLN
    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_cLN(v, T=self.T, axis=self.axis)
