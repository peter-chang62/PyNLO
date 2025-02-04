# %% ----- imports
from edf import utility
import pathlib
import pandas as pd
from scipy.constants import c
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

path = pathlib.Path(utility.__file__).parent


# %% ----- absorption and emission cross-sections
class crossSection:
    def __init__(self):
        sigma = pd.read_excel(
            path / "NLight_provided/Erbium Cross Section - nlight_pump+signal.xlsx"
        )
        sigma = sigma.to_numpy()[1:].astype(float)[:, [0, 2, 3]]
        a = sigma[:, :2]
        e = sigma[:, [0, 2]]

        self.sigma_a = InterpolatedUnivariateSpline(
            c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros"
        )

        self.sigma_e = InterpolatedUnivariateSpline(
            c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros"
        )


# %% ---- fiber dispersion
class ER110_4_125_betas:
    def __init__(self):
        frame = pd.read_excel(
            path / "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
        )
        gvd = frame.to_numpy()[:, :2][1:].astype(float)

        wl = gvd[:, 0] * 1e-9
        omega = 2 * np.pi * c / wl
        omega0 = 2 * np.pi * c / 1560e-9
        polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
        self.polyfit = polyfit[::-1]  # lowest order first


class ER80_4_125_betas:
    def __init__(self):
        frame = pd.read_excel(
            path
            / "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
        )
        gvd = frame.to_numpy()[:, :2][1:].astype(float)

        wl = gvd[:, 0] * 1e-9
        omega = 2 * np.pi * c / wl
        omega0 = 2 * np.pi * c / 1560e-9
        polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
        self.polyfit = polyfit[::-1]  # lowest order first


class ER80_8_125_betas:
    def __init__(self):
        frame = pd.read_excel(
            path / "NLight_provided/nLIGHT_Er80-8_125-PM_simulated_GVD_dispersion.xlsx"
        )
        gvd = frame.to_numpy()[:, :2][1:].astype(float)

        wl = gvd[:, 0] * 1e-9
        omega = 2 * np.pi * c / wl
        omega0 = 2 * np.pi * c / 1560e-9
        polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
        self.polyfit = polyfit[::-1]  # lowest order first
