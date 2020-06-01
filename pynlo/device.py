# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = []


# %% Imports

# %%
# gratting compressor

# prism compressor

# pulse shaper

# %% Old
# # %% Diffraction Grating Compressor ===========================================
# '''
# E. Treacy, "Optical pulse compression with diffraction gratings," in IEEE
# Journal of Quantum Electronics, vol. 5, no. 9, pp. 454-458, September 1969.
# doi: 10.1109/JQE.1969.1076303

# Jin Jer Huang, Liu Yang Zhang, and Yu Qiang Yang, "Reinvestigation on the
# frequency dispersion of a grating-pair compressor," Opt. Express 19, 814-819 (2011)
# https://doi.org/10.1364/OE.19.000814
# '''

# def grating_compressor_delay(v, phi1, d, s, ref_v=None, m=1, diagnostics=False):
#     """Group delay due to single pass between two diffraction gratings"""
#     #--- Diffraction
#     phi2 = np.arcsin(m*(c/v)/d - np.sin(phi1))

#     #--- Group Delay
#     delay = s/c*(1+np.sin(phi1)*np.sin(phi2))/np.cos(phi2)
#     if ref_v is not None:
#         ref_delay = grating_compressor_delay(ref_v, phi1, d, s, m=m)
#         delay = delay - ref_delay
#     return delay if (diagnostics==False) else (delay, phi2)

# def grating_compressor_phase(v, phi1, d, s, ref_v=None, m=1):
#     """Phase delay due to single pass between two diffraction gratings"""
#     #--- Diffraction
#     phi2 = np.arcsin(m*(c/v)/d - np.sin(phi1))

#     #--- Phase Delay
#     delay = s*(v/c)*np.cos(phi2)
#     if ref_v is not None:
#         ref_grp_delay = grating_compressor_delay(ref_v, phi1, d, s, m=m)
#         delay = delay - ref_grp_delay*(v - ref_v)
#     return delay

# plt.figure("grating compressor")
# plt.clf()
# # Thorlabs GTI25-03A
# #plt.plot(1e-12*v_grid, 1e12*np.gradient(grating_compressor_phase(v_grid, 31.7*pi/180, 1/300e3, test_length, ref_v=test_pump, m=1),df), label="test - phase")
# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 31.7*pi/180, 1/300e3, test_length, ref_v=test_pump, m=1), label="GTI25-03A")
# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 19*pi/180, 1/757e3, 2e-3, ref_v=test_pump, m=1), label="Ibsen FSTG-NIR757-905")

# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 64*pi/180, 1/1398.60e3, test_length, ref_v=test_pump, m=1), label="LS T-1400-800:64$^\circ$")
# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 54*pi/180, 1/1398.60e3, test_length, ref_v=test_pump, m=1), label="LS T-1400-800:54$^\circ$")
# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 44*pi/180, 1/1398.60e3, test_length, ref_v=test_pump, m=1), label="LS T-1400-800:44$^\circ$")
# # plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 34*pi/180, 1/1398.60e3, test_length, ref_v=test_pump, m=1), label="LS T-1400-800:34$^\circ$")

# test_length = 20e-3 * 2
# test_density = 75e3
# test_angle = 2. #38
# plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, test_angle*pi/180, 1/test_density, test_length, ref_v=test_pump, m=1), label="Test")

# #plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 31.7*pi/180, 1/300e3, 7e-3, ref_v=test_pump), label="300/mm,  7mm")
# #plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 31.7*pi/180, 1/200e3, 14e-3, ref_v=test_pump), label="200/mm, 14mm")
# #plt.plot(1e-12*v_grid, 1e12*grating_compressor_delay(v_grid, 31.7*pi/180, 1/100e3, 46.0e-3, ref_v=test_pump), label="100/mm, 46mm")
# #plt.plot(1e-12*v_grid, 1e12*test_delay)
# plt.grid(b=True)
# plt.ylim(-5, 5)
# #plt.ylim(-5, 100)
# plt.xlim(1e-12*v_grid.min(), 1e-12*v_grid.max())
# plt.ylabel("Group Delay (ps)")
# plt.xlabel("Frequency (THz)")
# plt.legend()
# plt.tight_layout()
# plt.xlim(190, 450)

# plt.figure("Grating Angle")
# plt.clf()
# plt.plot(v_grid*1e-12, test_angle*np.ones_like(v_grid))
# plt.plot(v_grid*1e-12, np.abs(grating_compressor_delay(v_grid, test_angle*pi/180, 1/test_density, test_length, ref_v=test_pump, m=1, diagnostics=True)[1] * 180/pi))
# plt.xlim(190, 450)
# # plt.xlim(270, 450)
# plt.grid(True)
# cursor = plt.matplotlib.widgets.Cursor(plt.gca(), useblit=True)



# # %% Prism Compressor =========================================================
# '''
# R. L. Fork, O. E. Martinez, and J. P. Gordon, "Negative dispersion using pairs
# of prisms," Opt. Lett. 9, 150-152 (1984)
# https://doi.org/10.1364/OL.9.000150

# R. E. Sherriff, "Analytic expressions for group-delay dispersion and cubic
# dispersion in arbitrary prism sequences," J. Opt. Soc. Am. B 15, 1224-1230 (1998)
# https://doi.org/10.1364/JOSAB.15.001224
# '''

# # def prism_compressor_delay(v, phi_1, alpha, s, delta, n, dndv, ref_v=None):
# #     """Group delay from single pass between two prisms"""
# #     #--- Refraction
# #     psi_1 = np.arcsin(np.sin(phi_1)/n(v))
# #     psi_2 = alpha - psi_1
# #     phi_2 = np.arcsin(n(v)*np.sin(psi_2))
# #     dphi2_dv = dndv(v)*np.sin(alpha)/((1-np.sin(phi_1)**2/n(v)**2)**0.5)/np.cos(phi_2)

# #     #--- Group Delay
# #     theta = delta - phi_2
# #     dtheta_dv = -dphi2_dv
# #     delay = s/c*(np.cos(theta) - v*np.sin(theta)*dtheta_dv)
# #     if ref_v is not None:
# #         ref_delay = prism_compressor_delay(ref_v, phi_1, alpha, s, delta, n, dndv)
# #         delay = delay - ref_delay
# #     return delay

# # def prism_compressor_phase(v, phi_1, alpha, s, delta, n, dndv, ref_v=None):
# #     """Phase delay from single pass between two prisms"""
# #     #--- Refraction
# #     psi_1 = np.arcsin(np.sin(phi_1)/n(v))
# #     psi_2 = alpha - psi_1
# #     phi_2 = np.arcsin(n(v)*np.sin(psi_2))

# #     #--- Phase Delay
# #     theta = delta - phi_2
# #     delay = s*(v/c)*np.cos(theta)
# #     if ref_v is not None:
# #         ref_grp_delay = prism_compressor_delay(ref_v, phi_1, alpha, s, delta, n, dndv)
# #         delay = delay - ref_grp_delay*(v - ref_v)
# #     return delay

# def prism_compressor_phase(v, phi_1, alpha, s, delta, n, ref_v=None):
#     """Phase delay from single pass between two prisms"""
#     #--- Refraction
#     psi_1 = np.arcsin(np.sin(phi_1)/n(v))
#     psi_2 = alpha - psi_1
#     phi_2 = np.arcsin(n(v)*np.sin(psi_2))

#     #--- Phase Delay
#     theta = delta - phi_2
#     phase_delay = np.nan_to_num(2*pi*s*(v/c)*np.cos(theta))
#     if ref_v is not None:
#         grp_delay = InterpolatedUnivariateSpline(v, phase_delay, k=3).derivative(1)
#         ref_grp_delay = grp_delay(ref_v)
#         phase_delay -= ref_grp_delay*(v - ref_v)
#     return phase_delay

# def prism_compressor_delay(v, phi_1, alpha, s, delta, n, ref_v=None):
#     """Group delay from single pass between two prisms"""
#     #--- Group Delay
#     phase_delay = prism_compressor_phase(v, phi_1, alpha, s, delta, n)
#     group_delay = InterpolatedUnivariateSpline(v, phase_delay/(2*pi), k=3).derivative(1)
#     if ref_v is not None:
#         ref_delay = group_delay(ref_v)
#     else:
#         ref_delay = 0
#     group_delay = group_delay(v) - ref_delay
#     return group_delay

# def prism_compressor_geometry(v, phi_1, alpha, s, delta, n, ref_v=None, w=2e-3):
#     """Incidence angles and beam magnifaction from single pass between two prisms"""
#     psi_1 = np.arcsin(np.sin(phi_1)/n(v))
#     psi_2 = alpha - psi_1
#     phi_2 = np.arcsin(n(v)*np.sin(psi_2))
#     theta = delta - phi_2

#     #--- 1st Prism
#     z_1_max = w/np.cos(phi_1) * np.cos(alpha/2) * np.ones_like(delay)
#     z_1_min = np.zeros_like(delay)
#     b_1 = 2*z_1_max*np.tan(alpha/2)
#     w_2 = w * np.cos(psi_1)/np.cos(phi_1) * np.cos(phi_2)/np.cos(psi_2)
#     #--- 2nd Prism
#     d_eff = s * np.cos(delta - alpha/2)
#     h_eff = d_eff * np.tan(delta - alpha/2)
#     beta = np.arcsin(d_eff/s)
#     gamma = pi - (-theta + beta + alpha/2)
#     x = s * np.sin(theta)/np.sin(gamma)
#     z_2_max = x * np.cos(alpha/2)
#     z_2_min = (x - w_2/np.cos(phi_2))*np.cos(alpha/2)
#     b_2 = 2*x*np.sin(alpha/2)

#     angles = {"phi_2":phi_2, "theta":theta, "beta":beta, "gamma":gamma}
#     return (z_1_min, z_1_max, b_1), (z_2_min, z_2_max, b_2), (d_eff, h_eff), angles

# def optimal_prism_angles(v, refractive_index, return_deg=True):
#     brewsters_angle = np.arctan(refractive_index(v))
#     internal_angle = np.arcsin(1/refractive_index(v)*np.sin(brewsters_angle))
#     apex_angle = 2*internal_angle
#     if return_deg:
#         brewsters_angle *= 180/pi
#         apex_angle *= 180/pi
#     return brewsters_angle, apex_angle

# def least_deviation_prism_angles(v, apex_angle, refractive_index, return_deg=True):
#     brewsters_angle = np.arctan(refractive_index(v))
#     internal_angle = apex_angle/2
#     incident_angle = np.arcsin(refractive_index(v)*np.sin(internal_angle))
#     if return_deg:
#         brewsters_angle *= 180/pi
#         incident_angle *= 180/pi
#     return brewsters_angle, incident_angle

# plt.figure("prism compressor")
# plt.clf()
# v_test = v_grid[(v_grid > 190e12) & (v_grid < 450e12)]
# #plt.plot(1e-12*v_test, 1e12*np.gradient(prism_compressor_phase(v_test, 55.60*pi/180, 69.1*pi/180, 10e-3, 60*pi/180, n_SiO2, dndv_SiO2_spline, ref_v=test_pump),df), label="test - phase")
# #plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 55.60*pi/180, 69.1*pi/180, 10e-3, 60*pi/180, n_SiO2, dndv_SiO2_spline, ref_v=test_pump), label="test - delay")

# # # Thorlabs AFS-CAF (~55.51 deg)
# # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 54.92*pi/180, 69.9*pi/180, test_length, test_delta, n_CaF2, dndv_CaF2_spline, ref_v=test_pump), label="AFS-CAF CaF$_2$")
# # # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 54.92*pi/180, 69.9*pi/180, -0.4, test_delta, n_CaF2, dndv_CaF2_spline, ref_v=test_pump), label="CaF$_2$: -40cm")
# # # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 54.92*pi/180, 69.9*pi/180, -0.8, test_delta, n_CaF2, dndv_CaF2_spline, ref_v=test_pump), label="CaF$_2$: -80cm")
# # # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 54.92*pi/180, 69.9*pi/180, -1.2, test_delta, n_CaF2, dndv_CaF2_spline, ref_v=test_pump), label="CaF$_2$: -120cm")

# # # Thorlabs AFS-FS (~56.65 deg)
# # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 55.30*pi/180, 69.1*pi/180, test_length, test_delta, n_SiO2, dndv_SiO2_spline, ref_v=test_pump), label="AFS-FS SiO$_2$")

# # # Newport 06LK10
# # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 58.25*pi/180, 63.0*pi/180, test_length, test_delta, n_N_LAK21, dndv_N_LAK21_spline, ref_v=test_pump), label="06LK10 N-LAK21")

# # Newport N-SF10
# prism_phi_1 = 59.7*pi/180 # Brewster at 800nm
# # prism_phi_1 = 59.18*pi/180 # Brewster at ~1060nm
# prism_alpha = 60.6*pi/180


# test_length = -7e-2 * 2
# test_delta = 70*pi/180 # 63.6
# delay = prism_compressor_delay(
#     v_test, prism_phi_1, prism_alpha, test_length, test_delta, n_N_SF10, ref_v=test_pump)
# (z_1_min, z_1_max, b_1), (z_2_min, z_2_max, b_2), (d_eff, h_eff), angles = prism_compressor_geometry(
#     v_test, prism_phi_1, prism_alpha, test_length, test_delta, n_N_SF10, ref_v=test_pump)

# test_length = 5e-2/2
# test_delta = 80. * pi/180 #66.3
# delay2 = prism_compressor_delay(
#     v_test, 2*prism_phi_1 - angles["phi_2"], prism_alpha, test_length, test_delta, n_N_SF10, ref_v=test_pump)
# (z2_1_min, z2_1_max, b2_1), (z2_2_min, z2_2_max, b2_2), (d2_eff, h2_eff), angles2 = prism_compressor_geometry(
#     v_test, 2*prism_phi_1 - angles["phi_2"], prism_alpha, test_length, test_delta, n_N_SF10, ref_v=test_pump)

# plt.plot(1e-12*v_test, 1e12*delay, label="10SF10 N-SF10")
# plt.plot(1e-12*v_test, 1e12*delay2, label="part 2")
# plt.plot(1e-12*v_test, 1e12*(delay + delay2), label="sum")


# # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, test_phi1, 60.6*pi/180, test_length, test_delta, n_N_SF10, dndv_N_SF10_spline, ref_v=test_pump), label="10SF10 N-SF10")
# # # Thorlabs AFS-SF14 (~61.0 deg)
# # #plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 59.44*pi/180, 59.6*pi/180, test_length, test_delta, n_N_SF14, dndv_N_SF14_spline, ref_v=test_pump), label="AFS-SF14 N-SF14")

# # # ZnSe
# # plt.plot(1e-12*v_test, 1e12*prism_compressor_delay(v_test, 68.06*pi/180, 43.88*pi/180, test_length, test_delta, n_ZnSe, dndv_ZnSe_spline, ref_v=test_pump), label="ZnSe")

# plt.grid(b=True)
# #plt.ylim(-5, 5)
# plt.ylim(-3, 3)
# plt.xlim(1e-12*v_test.min(), 1e-12*v_test.max())
# #plt.ylim(-5, 100)
# plt.title("Prism Pair")
# plt.ylabel("Group Delay (ps)")
# plt.xlabel("Frequency (THz)")
# plt.legend()
# plt.tight_layout()
# plt.xlim(190, 450)

# #print("Optimal Angles")
# #print(optimal_prism_angles(test_pump, n_CaF2))
# #print(optimal_prism_angles(test_pump, n_SiO2))
# #print(optimal_prism_angles(test_pump, n_N_SF10))
# #print(optimal_prism_angles(test_pump, n_N_SF14))
# #print(optimal_prism_angles(test_pump, n_N_LAK21))
# #print(optimal_prism_angles(test_pump, n_ZnSe))
# #
# #print("Least Deviation")
# #print(least_deviation_prism_angles(test_pump, 69.9*pi/180, n_CaF2))
# #print(least_deviation_prism_angles(test_pump, 69.1*pi/180, n_SiO2))
# #print(least_deviation_prism_angles(test_pump, 60.6*pi/180, n_N_SF10))
# #print(least_deviation_prism_angles(test_pump, 59.6*pi/180, n_N_SF14))
# #print(least_deviation_prism_angles(test_pump, 63.0*pi/180, n_N_LAK21))
# #print(least_deviation_prism_angles(test_pump, 60*pi/180, n_ZnSe))


# # %% Required Prism Dimensions

# separation = test_length
# w_d = 1e-3
# insertion_offset = 0#1.5e-3

# # # CaF
# # prism_delta = 56.7*pi/180 #56.5
# # prism_gamma = 54.92*pi/180
# # prism_alpha = 69.9*pi/180
# # delay, (z_1_min, z_1_max, b_1), (z_2_min, z_2_max, b_2), (d_eff, h_eff), angles = prism_compressor_delay(
# #     v_grid, prism_gamma, prism_alpha, separation, prism_delta,
# #     n_CaF2, dndv_CaF2_spline, ref_v=test_pump,
# #     return_diagnostics=True, w=w_d)

# # N_SF10
# prism_delta = test_delta # 60*pi/180 #63.5
# prism_gamma = prism_phi1 = 59.18*pi/180
# prism_alpha = 60.6*pi/180
# delay, (z_1_min, z_1_max, b_1), (z_2_min, z_2_max, b_2), (d_eff, h_eff), angles = prism_compressor_delay(
#     v_grid, prism_gamma, prism_alpha, separation, prism_delta,
#     n_N_SF10, dndv_N_SF10_spline, ref_v=test_pump,
#     return_diagnostics=True, w=w_d)

# plt.figure("test")
# plt.clf()
# plt.fill_between(1e-12*v_grid, insertion_offset + z_1_max, insertion_offset + z_1_min, label="z$_1$", alpha=0.75)
# plt.fill_between(1e-12*v_grid, -insertion_offset + z_2_max, -insertion_offset + z_2_min, label="z$_2$", alpha=0.75)
# plt.title("Prism Pair Insertion")
# plt.ylabel("Insertion Distance")
# plt.xlabel("Frequency (THz)")
# plt.legend()
# plt.xlim(190, 450)
# plt.ylim(bottom=0)
# plt.grid(True)

# def insertion_to_base(insertion, alpha=prism_alpha):
#     base = 2*insertion*np.tan(alpha/2)
#     return base

# def base_to_insertion(base, alpha=prism_alpha):
#     insertion = base/(2*np.tan(alpha/2))
#     return insertion

# ax2 = plt.gca().secondary_yaxis("right", functions=(insertion_to_base, base_to_insertion))
# ax2.set_ylabel("Minimum Base")
# plt.tight_layout()

# print(d_eff)
# print(h_eff)

# plt.figure("test2")
# plt.clf()
# plt.plot(1e-12*v_grid, 180/pi * prism_phi1*np.ones_like(v_grid), label=r"$\phi_1$")
# plt.plot(1e-12*v_grid, 180/pi * angles["phi_2"], label=r"$\phi_2$")
# plt.plot(1e-12*v_grid, 180/pi * prism_delta*np.ones_like(v_grid), label=r"$\delta$")
# plt.plot(1e-12*v_grid, 180/pi * angles["theta"], label=r"$\theta$")
# plt.plot(1e-12*v_grid, 180/pi * angles["beta"]*np.ones_like(v_grid), label=r"$\beta$")
# plt.plot(1e-12*v_grid, 180/pi * angles["gamma"], label=r"$\gamma$")
# plt.xlim(190, 450)
# plt.legend()
# plt.grid(True)

