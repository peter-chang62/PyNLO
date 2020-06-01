# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = []


# %% Imports


#%%

def n2_to_chi3():
    pass

def chi3_to_n2():
    pass

# %% Old
# %% Chi3 =====================================================================

# #--- Third Order Nonlinearity
# chi3 = 5200e-24

# #gamma3_pump_frq = (3/8)*(chi3 * 2*pi*pump_frq)/(c * n_eff(v=pump_frq) * A_eff(v=pump_frq))
# #g3_0 = (3/8)*(chi3/n_eff(z=0, v=v_grid))*(2*pi*v_grid/c)*(2/(e0*c*A_eff(z=0, v=v_grid)))
# g3_0 = (2*pi*v_grid*e0*chi3*A_eff(z=0, v=v_grid))/(e0*n_eff(z=0, v=v_grid)*c*A_eff(z=0, v=v_grid))**2
# def gamma3(z=0):
#     return g3_0
# gamma3 = None