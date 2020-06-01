# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = []


# %% Imports

# %% Old
# # %% Phase Matching ===========================================================
# def sfg_polling_period(v1, v2, m=1):
#     return np.abs(2*pi*m/(beta(v=v1) + beta(v=v2) - beta(v=v1+v2)))

# def dfg_polling_period(v1, v2, m=1):
#     """ v1 > v2"""
#     return np.abs(2*pi*m/(beta(v=v1) - beta(v=v2) - beta(v=v1+v2)))

# def qpm_coupling(m):
#     return np.abs(2/(m*pi)*np.sin(m*pi/2))

#  #---
# def l_c(self, v1, v2):
#     return 2/(self.beta(v1) - self.beta(v2))  # only good for chi2, coherence length

# def sfg_polling_period(self, v1, v2, m=1):
#     return np.abs(2*pi*m/(beta(v=v1) + beta(v=v2) - beta(v=v1+v2)))

# def dfg_polling_period(self, v1, v2, m=1):
#     """ v1 > v2"""
#     return np.abs(2*pi*m/(beta(v=v1) - beta(v=v2) - beta(v=v1+v2)))

# def qpm_coupling(m):
#     return np.abs(2/(m*pi)*np.sin(m*pi/2))


# # %% Phase Matching ===========================================================
# def sfg_polling_period(v1, v2, m=1):
#     return np.abs(2*pi*m/(beta(v=v1) + beta(v=v2) - beta(v=v1+v2)))

# def dfg_polling_period(v1, v2, m=1):
#     """ v1 > v2"""
#     return np.abs(2*pi*m/(beta(v=v1) - beta(v=v2) - beta(v=v1+v2)))

# def qpm_coupling(m):
#     return np.abs(2/(m*pi)*np.sin(m*pi/2))


# v_test = v_grid[::50]
# test_pump = c/1064e-9
# #test_pump = 276.5e12
# qpm_order = 1

# v_sum = np.add.outer(v_test, v_test)
# v_sum = np.ma.masked_where(v_sum>v_grid.max(), v_sum)

# v_dif = np.subtract.outer(v_test, v_test)
# v_dif = np.ma.masked_where(v_dif<v_grid.min(), v_dif)

# phase = beta(z=0, v=v_test)-2*pi*v_test*beta1(z=0, v=v_test)
# sfg_phase = beta(z=0, v=v_sum)-2*pi*v_sum*beta1(z=0, v=v_sum)
# dfg_phase = beta(z=0, v=v_dif)-2*pi*v_dif*beta1(z=0, v=v_dif)

# sfg_periods = 1e6*np.abs(2*pi*qpm_order/(beta(z=0, v=v_test[:,np.newaxis]) + beta(z=0, v=v_test[np.newaxis,:]) - beta(z=0, v=v_sum)))
# dfg_periods = 1e6*np.abs(2*pi*qpm_order/(beta(z=0, v=v_test[:, np.newaxis]) - beta(z=0, v=v_test[np.newaxis,:]) - beta(z=0, v=v_dif)))

# cmap = plt.cm.inferno_r
# cmap.set_bad(color='k')

# plt.figure("Grating Period - Sum Frequency")
# plt.clf()
# im = plt.imshow(sfg_periods, cmap=cmap, origin="lower", extent=(1e-12*v_test.min(), 1e-12*v_test.max(), 1e-12*v_test.min(), 1e-12*v_test.max()))
# cnt_sv = plt.contour(1e-12*v_test, 1e-12*v_test, 1e-12*v_sum, colors=".5", alpha=.25)
# cnt_sp = plt.contour(1e-12*v_test, 1e-12*v_test, sfg_periods, cmap=plt.cm.viridis)
# ax_lims = plt.xlim()
# plt.hlines(1e-12*test_pump, min(ax_lims), max(ax_lims), color='w', alpha=.5)
# ax_lims = plt.ylim()
# plt.vlines(1e-12*test_pump, min(ax_lims), max(ax_lims), color='w', alpha=.5)
# plt.colorbar(im, label=r"Optimal Grating Period ($\mu$m)")
# plt.clabel(cnt_sv, fmt="%.4g")
# plt.clabel(cnt_sp, fmt="%.4g")
# plt.title("Sum Frequency")
# plt.xlabel("+ Frequency (THz)")
# plt.ylabel("+ Frequency (THz)")
# plt.tight_layout()

# plt.figure("Grating Period - Dif Frequency")
# plt.clf()
# im = plt.imshow(dfg_periods, cmap=cmap, origin="lower", extent=(1e-12*v_test.min(), 1e-12*v_test.max(), 1e-12*v_test.min(), 1e-12*v_test.max()))
# cnt_dv = plt.contour(1e-12*v_test, 1e-12*v_test, 1e-12*v_dif, colors=".5", alpha=.25)
# cnt_dp = plt.contour(1e-12*v_test, 1e-12*v_test, dfg_periods, cmap=plt.cm.viridis)
# ax_lims = plt.xlim()
# plt.hlines(1e-12*test_pump, min(ax_lims), max(ax_lims), color='w', alpha=.5)
# ax_lims = plt.ylim()
# plt.vlines(1e-12*test_pump, min(ax_lims), max(ax_lims), color='w', alpha=.5)
# plt.colorbar(im, label=r"Optimal Grating Period ($\mu$m)")
# plt.clabel(cnt_dv, fmt="%.4g")
# plt.clabel(cnt_dp, fmt="%.4g")
# plt.title("Difference Frequency")
# plt.xlabel("- Frequency (THz)")
# plt.ylabel("+ Frequency (THz)")
# plt.tight_layout()

# # Group Velocity Mismatch
# plt.figure("Group Velocity Mismatch")
# plt.clf()
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1_CaF2(z=0, v=test_pump) - beta1_CaF2(z=0, v=v_grid)), label=r"CaF$_2$")
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1_SiO2(z=0, v=test_pump) - beta1_SiO2(z=0, v=v_grid)), label=r"SiO$_2$")
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1_N_LAK21(z=0, v=test_pump) - beta1_N_LAK21(z=0, v=v_grid)), label=r"N-LAK21")
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1_N_SF10(z=0, v=test_pump) - beta1_N_SF10(z=0, v=v_grid)), label=r"N-SF10")
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1_N_SF14(z=0, v=test_pump) - beta1_N_SF14(z=0, v=v_grid)), label=r"N-SF14")
# plt.plot(1e-12*v_grid, -1e12/1e3*(beta1(z=0, v=test_pump) - beta1(z=0, v=v_grid)), label=r"LN")
# plt.plot(1e-12*v_grid[v_grid < 750e12], -1e12/1e3*(beta1_ZnSe(z=0, v=test_pump) - beta1_ZnSe(z=0, v=v_grid))[v_grid < 750e12], label=r"ZnSe")
# plt.ylim(-1, 4.5)
# ax_lims = plt.ylim()
# plt.vlines(1e-12*test_pump, min(ax_lims), max(ax_lims), linestyle='--', label="{pump:.4g} THz".format(pump=1e-12*test_pump))
# plt.ylim(ax_lims)
# plt.xlabel('Frequency$_1$ (THz)')
# plt.ylabel("d$_{12}$ - Walk-Off Parameter (ps/mm)")
# plt.legend()
# plt.grid()
# plt.tight_layout()


# #%%

# #--- Interpolated Period
# z_periods_sample = np.linspace(0, length, int(1e5))
# periods_interp = InterpolatedUnivariateSpline(
#     test_z_sample,
#     test_periods)

# d_cycles_interp = InterpolatedUnivariateSpline(
#     z_periods_sample,
#     1/periods_interp(z_periods_sample))
# cycles_interp = d_cycles_interp.antiderivative(n=1)
# def cycles(z):
#     return cycles_interp(z) - cycles_interp(0)

# z_inversion = InterpolatedUnivariateSpline(
#     2*cycles(z_periods_sample),
#     z_periods_sample)

# z_invs = z_inversion(np.arange(int(cycles(length)*2)+1)) # all inversion points


# ##--- Constant Period
# #p_0 = 14e-6 #3.3e-6 #30e-6
# #def period(z):
# #    return p_0 # m
# #def cycles(z):
# #    '''integral(dz/period(z))'''
# #    return z/p_0
# #def z_inversion(n):
# #    '''the z value of the nth polling inversion point.'''
# #    return 0.5*n*p_0
# #z_invs = z_inversion(np.arange(int(cycles(length)*2)+1)) # all inversion points


# ##--- Linear Chirp
# #p_start =  13.0e-6 #13e-6 #periods_interp(0) #3.0e-6 #3.0e-6 #12.8e-6
# #p_stop = 23e-6 #2.0e-6 #periods_interp(z_stop) #6.75e-6 #6.75e-6 #19.4e-6
# #p_slope = (p_stop - p_start)/length # m/m
# #def period(z):
# #    return p_start + p_slope * z # m
# #def cycles(z):
# #    '''integral(dz/period(z))'''
# #    return np.log(1 + z * p_slope/p_start)/p_slope
# #def z_inversion(n):
# #    '''the z value of the nth polling inversion point.'''
# #    return (np.exp(n*p_slope/2)-1)*p_start/p_slope
# #z_invs = z_inversion(np.arange(int(cycles(length)*2)+1)) # all inversion points

# ## Random Linear
# ##z_diffs = np.diff(z_invs)
# ##np.random.shuffle(z_diffs) # an in place operation
# ##z_invs = np.cumsum([0] + z_diffs.tolist())


# ##--- Oscillating Linear Chirp
# #repetitions = 2
# #p_start = 12.8e-6
# #p_stop = 19.4e-6
# #p_slope = (p_stop - p_start)/(length/repetitions) # m/m
# #def period(z):
# #    return p_start + p_slope * z # m
# #def cycles(z):
# #    '''integral(dz/period(z))'''
# #    return np.log(1 + z * p_slope/p_start)/p_slope
# #def z_inversion(n):
# #    '''the z value of the nth polling inversion point.'''
# #    return (np.exp(n*p_slope/2)-1)*p_start/p_slope
# #z_diffs = np.diff(z_inversion(np.arange(int(cycles(length/repetitions)*2)+1)))
# ##z_invs = np.cumsum([0] + np.tile(z_diffs, repetitions).tolist())
# #z_invs = np.cumsum([0] + np.tile(z_diffs.tolist() + z_diffs[::-1].tolist(), repetitions//2).tolist())


# ##--- Quadratic Chirp
# #p_start = 19.4e-6
# #p_stop = 12.8e-6
# #p_vel = 3*(p_stop - p_start)/length # m/m
# #p_acc = 2*(p_stop - (p_start+p_vel*length))/length**2 # m/m**2
# #p_det = np.sqrt(2*p_start*p_acc - p_vel**2, dtype=complex)
# #def period(z):
# #    return p_start + p_vel * z + 0.5*p_acc*z**2 # m
# #def cycles(z):
# #    '''integral(dz/period(z))'''
# #    return np.real((2/p_det)*(np.arctan((p_vel + p_acc*z)/p_det) - np.arctan(p_vel/p_det)))
# #def z_inversion(n):
# #    '''the z value of the nth polling inversion point.'''
# #    return np.real((p_det*np.tan(n/4*p_det + np.arctan(p_vel/p_det)) - p_vel)/p_acc)
# #z_invs = z_inversion(np.arange(int(cycles(length)*2)+1)) # all inversion points


# # Plotting
# plt.figure("Gratings")
# plt.clf()
# plt.plot(1e3*z_invs[1:], 1e6*2*np.diff(z_invs))
# plt.grid()
# plt.xlabel("Propagation Distance (mm)")
# plt.ylabel("Grating Period ($\mu$m)")
# plt.title("Grating Profile")
# plt.tight_layout()

# #--- Poling Sign
# #z_invs = z_invs - 0.5*(z_invs[1] - z_invs[0]) # shift by 1/4 phase
# #z_invs[0] = 0
# def pol_sign(z):
#     return (-1)**(np.count_nonzero(np.logical_not(z < z_invs)) % 2)

# #--- Second Order Nonlinearity
# deff = 23e-12 # 19.6e-12 # m/v
# chi2 = 2*deff

# #chi2 = (n_eff_spline(v_grid)**2 - 1) * 2*23e-12/(n_eff_spline(c/1.1e-6)**2 - 1) #doesn't conserve energy

# #gamma2_pump_frq = (1/4)*(chi2/n_eff(v=pump_frq))*(2*pi*pump_frq/c)*np.sqrt(2/(e0*c*A_eff(v=pump_frq)))
# #g2_0 = (chi2/n_eff(z=0, v=v_grid))*(2*pi*v_grid/c)*np.sqrt(2/(e0*c*A_eff(z=0, v=v_grid)))
# #g2_0 = (1/4)*(chi2/n_eff(z=0, v=group_frq))*(2*pi*v_grid/c)*np.sqrt(2/(e0*c*A_eff(z=0, v=group_frq)))
# g2_0 = (2*pi*v_grid*e0*chi2*A_eff(z=0, v=v_grid))/(e0*n_eff(z=0, v=v_grid)*c*A_eff(z=0, v=v_grid))**(3/2)
# # g2_0 *= 10
# def gamma2(z=0):
#     return g2_0 * pol_sign(z)