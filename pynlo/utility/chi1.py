# -*- coding: utf-8 -*-
"""
TODO: module docs
"""

__all__ = []


# %% Imports


#%%

def d_12(self, beta1, beta2):
    # if single values
    # if signal, array
    # if array array
    return self.beta_1(v1) - self.beta_1(v2)



# %% Old
# import numpy as np
# from scipy.constants import pi, c
# # expand beta coefs about some freq., etc...

# #%% TEMP

# def n_SiO2(v):
#     '''
#     I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused
#     Silica*,†," J. Opt. Soc. Am. 55, 1205-1209 (1965)
#     https://doi.org/10.1364/JOSA.55.001205
#     '''
#     x = c/v * 1e6
#     n = (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5
#     return n

# # %% Fresnel Loss
# v_ref = c/1700e-9
# v_ref = c/1350e-9
# v_ref = c/1060e-9
# v_ref = c/650e-9

# n1 = 1
# th1 = 55.286 * pi/180
# # th1 = 0.5*7.1 * pi/180
# # th1 = 0.5*2.8 * pi/180

# n2 = 1.4434
# n2 = n_SiO2(v_ref)
# th2 = np.arcsin(n1/n2 * np.sin(th1))

# print(th2*180/pi)

# r = np.abs((n1*np.cos(th2) - n2*np.cos(th1))/(n1*np.cos(th2) + n2*np.cos(th1)))**2
# t = 1-r

# print(t)

# n1 = n2
# th1 = 69.06*pi/180 - th2
# # th1 = 7.1*pi/180 - th2
# # th1 = 2.8*pi/180 - th2
# n2 = 1
# th2 = np.arcsin(n1/n2 * np.sin(th1))

# r = np.abs((n1*np.cos(th2) - n2*np.cos(th1))/(n1*np.cos(th2) + n2*np.cos(th1)))**2

# print(1-r)
# print(t*(1-r))

# #%%


