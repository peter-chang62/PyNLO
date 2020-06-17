# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = ["Medium", "Waveguide"]


# %% Imports

import functools

import numpy as np
from scipy import constants

# %% Constants

pi = constants.pi
c = constants.c
e0 = constants.epsilon_0


# %% Base Classes
class Medium():
    def __init__(self, v_grid, n_v, alpha_v=None):
        self._v_grid = np.asarray(v_grid)

        #--- Refractive Index
        if callable(n_v):
            self._n = n_v
        else:
            assert (len(n_v) == len(v_grid)), "The length of n_v must match v_grid."
            self._n = np.asarray(n_v)

        #--- Loss
        if callable(alpha_v):
            self._alpha = alpha_v
        elif alpha_v is not None:
            assert (len(alpha_v) == len(v_grid)), "The length of alpha_v must match v_grid."
            self._alpha = np.asarray(alpha_v)
        else:
            self._alpha = np.zeros_like(self.v_grid)

        #--- Cached Functions
        self.dndv = functools.lru_cache()(self.dndv)
        self.beta = functools.lru_cache()(self.beta)

    @property
    def v_grid(self):
        """
        The frequency grid.

        Returns
        -------
        array of float
        """
        return self._v_grid

    @property
    def n(self):
        """
        The refractive index.

        Returns
        -------
        array of float
        """
        if callable(self._n):
            self._n = self._n(self.v_grid)
        return self._n

    @property
    def alpha(self):
        """
        The attenuation constant.

        Positive values correspond to loss and negative values corespond to
        gain.

        Returns
        -------
        array of float
        """
        if callable(self._alpha):
            self._alpha = self._alpha(self.v_grid)
        return self._alpha

    @property
    def chi_1(self):
        """
        The linear susceptibility.

        Returns
        -------
        array of complex
        """
        return (self.n + (0.5j*c/(2*pi))*self.alpha/self.v_grid)**2 - 1

    def chi_2(self, *args, **kwargs):
        """
        The second order nonlinear optical susceptibility.

        Derived classes should override this method with useful data.

        Returns
        -------
        NotImplementedError
        """
        return NotImplementedError

    def chi_3(self, *args, **kwargs):
        """
        The third order nonlinear optical susceptibility.

        Derived classes should override this method with useful data.

        Returns
        -------
        NotImplementedError
        """
        return NotImplementedError

    def dndv(self, m=1):
        """
        The derivative of the refractive index with repsect to frequency.

        This method only returns the refractive index unchanged when `m` is
        less than or equal to 0.

        Parameters
        ----------
        m : int, optional
            The order of the returned derivative. The default is 1.

        Returns
        -------
        array of float
        """
        assert isinstance(m, (int, np.integer)), "The derivative order must be an integer."

        #--- Calculate Derivatives
        if m <= 0:
            dndv = self.n
        else:
            dndv = np.gradient(self.dndv(m=m-1), self.v_grid)
        return dndv

    def beta(self, m=0):
        """
        The propagation constant.

        Parameters
        ----------
        m : int, optional
            The order of the returned derivative of the propagation constant.
            The default is 0, which returns the propagation constant without
            taking the derivative.

        Returns
        -------
        array of float
        """
        return 1/c * (2*pi)**(1-m) * (m*self.dndv(m=m-1) + self.v_grid*self.dndv(m=m))

    def n_g(self):
        """
        Group Index

        Returns
        -------
        array of float
        """
        return c*self.beta(n=1)

    def v_g(self):
        """
        Group velocity

        Returns
        -------
        array of float
        """
        return 1/self.beta(n=1)

    def GVD(self):
        """
        Group velocity dispersion

        Returns
        -------
        array of float
        """
        return self.beta(n=2)

    def D(self):
        """
        Dispersion parameter

        Returns
        -------
        array of float
        """
        return -2*pi/c * self.v_grid**2 * self.beta(n=2)

    def linear_operator(self, dz, v_0=None):
        #--- Phase and Loss
        phase = dz*self.beta()
        loss = 0.5j*dz*self.alpha

        #--- Comoving Reference Frame
        if v_0 is None:
            v_0_idx = len(self.v_grid)//2
        else:
            v_0_idx = np.armin(np.abs(v_0 - self.v_grid))
        phase_ret = 2*pi*dz*self.beta(n=1)[v_0_idx]*self.v_grid

        #--- Linear Operator
        operator = np.exp(1j*((phase - phase_ret) + loss))
        return operator, phase, phase_ret, loss #TODO: namedcollection?

# def try_z(func):
#     try:
#         return func(self.v_grid, self.z)
#     except TypeError:
#         return func(self.v_grid)

class Waveguide(Medium):
    """
    effective values, integrated over the waveguide mode.
    """
    def __init__(self, v_grid, n_v, alpha_v=None, a_eff=None, g2_v=None, g3_v=None, z=None):
        #--- Refractive Index and Loss
        super().__init__(v_grid, n_v, alpha_v=alpha_v)

        #--- Effective Area

        #--- Second Order Nonlinearity
        if isinstance(g2_v, dict):
            # Effective Second Order Optical Nonlinearity
            self._chi2 = g2_v["chi2_v"] if "chi2_v" in g2_v else None
        elif g2_v is not None:
            g2_0, v_0 = g2_v

        #--- Third Order Nonlinearity
        if isinstance(g3_v, dict):
            # Effective Third Order Optical Nonlinearity
            self._chi3 = g3_v["chi3_v"] if "chi3_v" in g3_v else None
            # Effective Ramman Response
            self._r = g3_v["r_v"] if "r_v" in g3_v else None
            # Effective Ramman Fraction
            self._rf = g3_v["rf"] if "r3_v" in g3_v else None
        elif g3_v is not None:
            g3_0, v_0 = g3_v

    @property
    def n(self):
        """
        The refractive index.

        Returns
        -------
        array of float
        """
        if callable(self._n):
            if self.z is None:
                return self._n(self.v_grid)
            else:
                return self._n(self.v_grid, self.z)
        else:
            return self._n

    @property
    def alpha(self):
        """
        The attenuation constant.

        Positive values correspond to loss and negative values corespond to
        gain.

        Returns
        -------
        array of float
        """
        if callable(self._alpha):
            if self.z is None:
                return self._alpha(self.v_grid)
            else:
                return self._alpha(self.v_grid, self.z)
        else:
            return self._alpha

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, z):
        self.dndv.cache_clear()
        self.beta.cache_clear()
        self._z = z



# %% Bulk Materials

class SiO2(Medium):
    def __init__(self, v_grid):
        self._generating_function = None

    @classmethod
    def alternate_source(self, v_grid):
        pass

class MgLN(Medium):
    def __init__(self, v_grid, axis="e", T=25):
        pass





# # %% Waveguide


# class Waveguide():
#     pass # can be z dependent


# #%% Bulk Refractive Indicies


# #--- Fused Silica
# def n_SiO2(v):
#     '''
#     I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused
#     Silica*,†," J. Opt. Soc. Am. 55, 1205-1209 (1965)
#     https://doi.org/10.1364/JOSA.55.001205
#     '''
#     x = c/v * 1e6
#     n = (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5
#     return n
# n_SiO2_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     n_SiO2(v_grid),
#     k=3)
# dndv_SiO2_spline = n_SiO2_spline.derivative(n=1)
# def beta_SiO2(z=0, v=ref_frq):
#     return n_SiO2(v) * 2*pi*v/c
# def beta1_SiO2(z=0, v=ref_frq):
#     return (n_SiO2(v) + v*dndv_SiO2_spline(v))/c

# #--- Calcium Flouride
# def n_CaF2(v):
#     '''
#     Li, H. H. "Refractive index of alkaline earth halides and its wavelength
#     and temperature derivatives." Journal of Physical and Chemical Reference
#     Data 9.1 (1980): 161-290.
#     https://doi.org/10.1063/1.555616
#     '''
#     x = c/v * 1e6
#     n = (1+0.33973+0.69913/(1-(0.09374/x)**2)+0.11994/(1-(21.18/x)**2)+4.35181/(1-(38.46/x)**2))**.5
#     return n
# n_CaF2_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     n_CaF2(v_grid),
#     k=3)
# dndv_CaF2_spline = n_CaF2_spline.derivative(n=1)
# def beta_CaF2(z=0, v=ref_frq):
#     return n_CaF2(v) * 2*pi*v/c
# def beta1_CaF2(z=0, v=ref_frq):
#     return (n_CaF2(v) + v*dndv_CaF2_spline(v))/c

# #--- SCHOTT N-SF10
# def n_N_SF10(v):
#     '''
#     https://refractiveindex.info/?shelf=glass&book=SF10&page=SCHOTT
#     '''
#     x = c/v * 1e6
#     n=(1+1.62153902/(1-0.0122241457/x**2)+0.256287842/(1-0.0595736775/x**2)+1.64447552/(1-147.468793/x**2))**.5
#     return n
# n_N_SF10_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     n_N_SF10(v_grid),
#     k=3)
# dndv_N_SF10_spline = n_N_SF10_spline.derivative(n=1)
# def beta_N_SF10(z=0, v=ref_frq):
#     return n_N_SF10(v) * 2*pi*v/c
# def beta1_N_SF10(z=0, v=ref_frq):
#     return (n_N_SF10(v) + v*dndv_N_SF10_spline(v))/c

# #--- SCHOTT N-SF14
# def n_N_SF14(v):
#     '''
#     https://refractiveindex.info/?shelf=glass&book=SCHOTT-SF&page=N-SF14
#     '''
#     x = c/v * 1e6
#     n=(1+1.69022361/(1-0.0130512113/x**2)+0.288870052/(1-0.061369188/x**2)+1.7045187/(1-149.517689/x**2))**.5
#     return n
# n_N_SF14_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     n_N_SF14(v_grid),
#     k=3)
# dndv_N_SF14_spline = n_N_SF14_spline.derivative(n=1)
# def beta_N_SF14(z=0, v=ref_frq):
#     return n_N_SF14(v) * 2*pi*v/c
# def beta1_N_SF14(z=0, v=ref_frq):
#     return (n_N_SF14(v) + v*dndv_N_SF14_spline(v))/c

# #--- SCHOTT N-LAK21
# def n_N_LAK21(v):
#     '''
#     https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaK&page=N-LAK21
#     '''
#     x = c/v * 1e6
#     n=(1+1.22718116/(1-0.00602075682/x**2)+0.420783743/(1-0.0196862889/x**2)+1.01284843/(1-88.4370099/x**2))**.5
#     return n
# n_N_LAK21_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     n_N_LAK21(v_grid),
#     k=3)
# dndv_N_LAK21_spline = n_N_LAK21_spline.derivative(n=1)
# def beta_N_LAK21(z=0, v=ref_frq):
#     return n_N_LAK21(v) * 2*pi*v/c
# def beta1_N_LAK21(z=0, v=ref_frq):
#     return (n_N_LAK21(v) + v*dndv_N_LAK21_spline(v))/c

# #--- Zinc Selenide
# def n_ZnSe(v):
#     '''
#     J. Connolly, B. diBenedetto, R. Donadio, "Specifications Of Raytran
#     Material," Proc. SPIE 0181, Contemporary Optical Systems and Components
#     Specifications, (7 September 1979)
#     https://doi.org/10.1117/12.957359

#     Berge Tatian, "Fitting refractive-index data with the Sellmeier dispersion
#     formula," Appl. Opt. 23, 4477-4485 (1984)
#     https://doi.org/10.1364/AO.23.004477
#     '''
#     x = c/v * 1e6
#     n=(1+4.45813734/(1-(0.200859853/x)**2)+0.467216334/(1-(0.391371166/x)**2)+2.89566290/(1-(47.1362108/x)**2))**.5
#     return n
# n_ZnSe_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     np.nan_to_num(n_ZnSe(v_grid)),
#     k=3)
# dndv_ZnSe_spline = n_ZnSe_spline.derivative(n=1)
# def beta_ZnSe(z=0, v=ref_frq):
#     return n_ZnSe(v) * 2*pi*v/c
# def beta1_ZnSe(z=0, v=ref_frq):
#     return (n_ZnSe(v) + v*dndv_ZnSe_spline(v))/c

# #--- Zn-Doped Lithium Niobate
# def n_ZnLN_S(v, T=24.5, axis="e", c_Zn=6.5, c_Li=48.5):
#     """
#     400 to 1200nm
#     20 to 100 C
#     47 to 50 mol % LiO
#     <8 mol % Zn

#     Schlarb, U., et al. "Refractive indices of Zn-doped lithium niobate."
#     Optical Materials 4.6 (1995): 791-795.
#     https://doi.org/10.1016/0925-3467(95)00018-6
#     """
#     if axis=="e":
#         wvl0 = 218.203 # pole in UV
#         u0 = 6.4047e-6
#         a0 = 3.9466e-5
#         a_Nb = 11.8635e-7
#         a_Zn = 1.9221e-7
#         a_IR = 3.0998e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV
#     elif axis=="o":
#         wvl0 = 223.219 # pole in UV
#         u0 = 1.1082e-6
#         a0 = 4.5312e-5
#         a_Nb = -7.2320e-8
#         a_Zn = 6.7963e-8
#         a_IR = 3.6340e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV

#     f = (T + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(T+273)) - 1)
#     f0 = (24.5 + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(24.5+273)) - 1)
#     F = f - f0

#     c_thr = 6.5*(2/3)*(50 - c_Li)
#     if c_Zn < c_thr:
#         c_Nb = (2/3)*(50 - c_Li) - c_Zn/6.5
#     else:
#         c_Nb = 0
#     wvl_T = wvl0 + u0*F

#     wvl = c/v * 1e9 # nm
#     n2 = (a0 + a_Nb*c_Nb + a_Zn*c_Zn)/(wvl_T**-2 - wvl**-2) - a_IR*wvl**2 + a_UV
#     return n2**0.5

# #--- Mg-Doped Lithium Niobate
# def n_MgLN_SB(v, T=24.5, axis="e", c_Mg=5, c_Li=48.5):
#     """
#     400 to 1200nm
#     -50 to 250 C
#     47 to 50 mol % LiO
#     <9 mol % Mg

#     Schlarb, U., and K. Betzler. "Influence of the defect structure on the
#     refractive indices of undoped and Mg-doped lithium niobate." Physical
#     Review B 50.2 (1994): 751.
#     https://doi.org/10.1103/PhysRevB.50.751
#     """
#     if axis=="e":
#         wvl0 = 218.203 # pole in UV
#         u0 = 6.4047e-6
#         a0 = 3.9466e-5
#         a_Nb = 2.3727e-7
#         a_Mg = 7.6243e-8
#         a_IR = 3.0998e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV
#     elif axis=="o":
#         wvl0 = 223.219 # pole in UV
#         u0 = 1.1082e-6
#         a0 = 4.5312e-5
#         a_Nb = -1.4464e-8
#         a_Mg = -7.3548e-8
#         a_IR = 3.6340e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV

#     f = (T + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(T+273)) - 1)
#     f0 = (24.5 + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(24.5+273)) - 1)
#     c_thr = (10/3)*(50 - c_Li)
#     if c_Mg < c_thr:
#         a = a0 + (c_thr - c_Mg)*a_Nb + c_Mg*a_Mg
#     else:
#         a = a0 + c_Mg*a_Mg
#     wvl_T = wvl0 + u0*(f - f0)

#     wvl = c/v * 1e9 # nm
#     n2 = a/(wvl_T**-2 - wvl**-2) - a_IR*wvl**2 + a_UV
#     return n2**0.5

# def n_MgLN_G(v, T=24.5, axis="e"):
#     """
#     500 to 4000nm
#     20 to 200 C
#     48.5 mol % Li
#     5 mol % Mg


#     Gayer, O., et al. "Temperature and wavelength dependent refractive index
#     equations for MgO-doped congruent and stoichiometric LiNbO 3." Applied
#     Physics B 91.2 (2008): 343-348.
#     http://dx.doi.org/10.1007/s00340-010-4203-7
#     """
#     if axis=="e":
#         a1 = 5.756 # plasmons in the far UV
#         a2 = 0.0983 # weight of UV pole
#         a3 = 0.2020 # pole in UV
#         a4 = 189.32 # weight of IR pole
#         a5 = 12.52 # pole in IR
#         a6 = 1.32e-2 # phonon absorption in IR
#         b1 = 2.860e-6
#         b2 = 4.700e-8
#         b3 = 6.113e-8
#         b4 = 1.516e-4
#     elif axis=="o":
#         a1 = 5.653 # plasmons in the far UV
#         a2 = 0.1185 # weight of UV pole
#         a3 = 0.2091 # pole in UV
#         a4 = 89.61 # weight of IR pole
#         a5 = 10.85 # pole in IR
#         a6 = 1.97e-2 # phonon absorption in IR
#         b1 = 7.941e-7
#         b2 = 3.134e-8
#         b3 = -4.641e-9
#         b4 = -2.188e-6

#     wvl = c/v * 1e6 # um
#     f = (T-24.5)*(T+570.82)
#     n2 = (a1 + b1*f) + (a2 + b2*f)/(wvl**2 - (a3 + b3*f)**2) + (a4 + b4*f)/(wvl**2 - a5**2) - a6*wvl**2
#     return n2**0.5

# #--- X-Doped Lithium Niobate
# def n_XLN_G(v, x_UV=1., x_IR=1., T=24.5, axis="e"):
#     """
#     An approximation... the resonant frequencies of the Nb, Mg, and Zn on the
#     lithium sites are take to be approximately equal... The only significant difference is the strength of the oscillator,
#     thus by adjusting the oscillator strengths ... one can fit to any concentration of Mg and Zn doping...

#     Schlarb, U., and K. Betzler. "Influence of the defect structure on the
#     refractive indices of undoped and Mg-doped lithium niobate." Physical
#     Review B 50.2 (1994): 751.
#     https://doi.org/10.1103/PhysRevB.50.751

#     Schlarb, U., et al. "Refractive indices of Zn-doped lithium niobate."
#     Optical Materials 4.6 (1995): 791-795.
#     https://doi.org/10.1016/0925-3467(95)00018-6


#     Based on the Gayer data:
#     500 to 4000nm
#     20 to 200 C
#     48.5 mol % Li
#     5 mol % Mg

#     Gayer, O., et al. "Temperature and wavelength dependent refractive index
#     equations for MgO-doped congruent and stoichiometric LiNbO 3." Applied
#     Physics B 91.2 (2008): 343-348.
#     http://dx.doi.org/10.1007/s00340-010-4203-7
#     """
#     if axis=="e":
#         a1 = 5.756 # plasmons in the far UV
#         a2 = 0.0983 # weight of UV pole
#         a3 = 0.2020 # pole in UV
#         a4 = 189.32 # weight of IR pole
#         a5 = 12.52 # pole in IR
#         a6 = 1.32e-2 # phonon absorption in IR
#         b1 = 2.860e-6
#         b2 = 4.700e-8
#         b3 = 6.113e-8
#         b4 = 1.516e-4
#     elif axis=="o":
#         a1 = 5.653 # plasmons in the far UV
#         a2 = 0.1185 # weight of UV pole
#         a3 = 0.2091 # pole in UV
#         a4 = 89.61 # weight of IR pole
#         a5 = 10.85 # pole in IR
#         a6 = 1.97e-2 # phonon absorption in IR
#         b1 = 7.941e-7
#         b2 = 3.134e-8
#         b3 = -4.641e-9
#         b4 = -2.188e-6

#     wvl = c/v * 1e6 # um
#     f = (T-24.5)*(T+570.82)
#     n2 = (a1 + b1*f) + x_UV*(a2 + b2*f)/(wvl**2 - (a3 + b3*f)**2) + x_IR*(a4 + b4*f)/(wvl**2 - a5**2) - a6*wvl**2
#     return n2**0.5

# def n_XLN_SB(v, x_UV=1, T=24.5, axis="e", c_Mg=5, c_Li=48.5):
#     """
#     400 to 1200nm
#     -50 to 250 C
#     47 to 50 mol % LiO
#     <9 mol % Mg

#     Schlarb, U., and K. Betzler. "Influence of the defect structure on the
#     refractive indices of undoped and Mg-doped lithium niobate." Physical
#     Review B 50.2 (1994): 751.
#     https://doi.org/10.1103/PhysRevB.50.751
#     """
#     if axis=="e":
#         wvl0 = 218.203 # pole in UV
#         u0 = 6.4047e-6
#         a0 = 3.9466e-5
#         a_Nb = 2.3727e-7
#         a_Mg = 7.6243e-8
#         a_IR = 3.0998e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV
#     elif axis=="o":
#         wvl0 = 223.219 # pole in UV
#         u0 = 1.1082e-6
#         a0 = 4.5312e-5
#         a_Nb = -1.4464e-8
#         a_Mg = -7.3548e-8
#         a_IR = 3.6340e-8 # phonon absorption in IR
#         a_UV = 2.6613 # plasmons in the far UV

#     f = (T + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(T+273)) - 1)
#     f0 = (24.5 + 273)**2 + 4.0238e5*(1/np.tanh(261.6/(24.5+273)) - 1)
#     c_thr = (10/3)*(50 - c_Li)
#     if c_Mg < c_thr:
#         a = a0 + (c_thr - c_Mg)*a_Nb + c_Mg*a_Mg
#     else:
#         a = a0 + c_Mg*a_Mg
#     wvl_T = wvl0 + u0*(f - f0)

#     wvl = c/v * 1e9 # nm
#     n2 = x_UV*a/(wvl_T**-2 - wvl**-2) - a_IR*wvl**2 + a_UV
#     return n2**0.5


# # %% Waveguide Geometry, Dispersion, and Loss =================================

# eff_area = 15e-6 * 16e-6 # m**2


# #--- Effective Area -----------------------------------------------------------
# def A_eff(z=0, v=ref_frq):
#     return eff_area

# #--- Effective Index ----------------------------------------------------------
# def n_eff(z=0, v=ref_frq):
#     # return n_MgLN_G(v, T=24.5, axis="e")
#     return n_XLN_G(v, T=24.5, x_UV=1.061)
# n_eff_spline = InterpolatedUnivariateSpline(
#     v_grid[v_grid > 50e12],
#     n_eff(v=v_grid[v_grid > 50e12]),
#     ext="extrapolate")
# dndv_spline = n_eff_spline.derivative(1)
# d2ndv2_spline = n_eff_spline.derivative(2)

# #--- Propagation Constant -----------------------------------------------------
# def beta(z=0, v=ref_frq):
#     return n_eff_spline(v) * 2*pi*v/c

# def beta1(z=0, v=ref_frq):
#     return (n_eff_spline(v) + v*dndv_spline(v))/c

# def beta2(z=0, v=ref_frq):
#     return (2*dndv_spline(v) + v*d2ndv2_spline(v))/(2*pi*c)

# def D(z=0, v=ref_frq):
#     return -(d2ndv2_spline(v)*(v**2/c)**2 + dndv_spline(v)*(2*v**3/c**2))/v

# #--- Gain/Loss ----------------------------------------------------------------
# #alpha = 10*(1+special.erf(-(v_grid-350.)/(10*np.sqrt(2)))) # 10 /m roughly at 5.5 microns according to Schwesyg et al.
# alpha_dB = 0 # dB/m
# def alpha(z=0, v=ref_frq):
#     return alpha_dB * np.log(10)/10 # in linear units

# ##--- Plotting -----------------------------------------------------------------

# def test_n(v):
#     x = c/v * 1e6
#     n=(1+2.9804/(1-0.02047/x**2)+0.5981/(1-0.0666/x**2)+8.9543/(1-416.08/x**2))**.5 # e
#     #n=(1+2.6734/(1-0.01764/x**2)+1.2290/(1-0.05914/x**2)+12.614/(1-474.60/x**2))**.5 # o
#     return n
# test_n_spline = InterpolatedUnivariateSpline(
#     v_grid,
#     test_n(v_grid),
#     k=3)
# test_dndv_spline = test_n_spline.derivative(n=1)
# def test_beta(z=0, v=ref_frq):
#     return 1.02*test_n(v) * 2*pi*v/c
# def test_beta1(z=0, v=ref_frq):
#     return 1.02*(test_n(v) + v*test_dndv_spline(v))/c





# # %%