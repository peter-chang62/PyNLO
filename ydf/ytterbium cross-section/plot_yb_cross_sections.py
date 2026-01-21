#!/usr/bin/env python3
"""
Plot Ytterbium cross-sections as 2D heatmaps and 1D plots for specific temperatures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import clipboard

# Load the data - read header rows separately
absorption_df = pd.read_csv('absorption cross section.csv', skiprows=2)
emission_df = pd.read_csv('emission cross section.csv', skiprows=2)

# Read temperature header (row 2, 0-indexed)
temp_header = pd.read_csv('absorption cross section.csv', nrows=1, skiprows=2, header=None)
temp_values = temp_header.iloc[0, 1:].values  # Skip first column (wavelength label)
temperatures = np.array([float(str(val).split()[0]) for val in temp_values])

# Extract wavelengths
wavelengths_abs = absorption_df.iloc[:, 0].values
wavelengths_em = emission_df.iloc[:, 0].values

# Extract cross-section data
absorption_data = absorption_df.iloc[:, 1:].values
emission_data = emission_df.iloc[:, 1:].values

# Find indices for specific temperatures
idx_77K = np.where(temperatures == 77)[0][0]
idx_300K = np.where(temperatures == 300)[0][0]
idx_420K = np.where(temperatures == 420)[0][0]

# ============================================================================
# Figure 1: 2D Heatmaps
# ============================================================================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 2D Heatmap - Absorption
im1 = ax1.pcolormesh(wavelengths_abs, temperatures, absorption_data.T,
                      shading='auto', cmap='hot')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Temperature (K)')
ax1.set_title('Ytterbium Absorption Cross-section: Temperature vs Wavelength')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Absorption Cross-section (m²)')
ax1.grid(True, alpha=0.3)

# 2D Heatmap - Emission
im2 = ax2.pcolormesh(wavelengths_em, temperatures, emission_data.T,
                      shading='auto', cmap='plasma')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Temperature (K)')
ax2.set_title('Ytterbium Emission Cross-section: Temperature vs Wavelength')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Emission Cross-section (m²)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the heatmap figure
plt.savefig('yb_cross_sections_heatmaps.png', dpi=300, bbox_inches='tight')
print("Heatmap plot saved as 'yb_cross_sections_heatmaps.png'")

# ============================================================================
# Figure 2: 1D Plots at Selected Temperatures
# ============================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

# 1D Plots - Absorption for 77K, 300K, 420K
ax3.plot(wavelengths_abs, absorption_data[:, idx_77K], label='77 K')
ax3.plot(wavelengths_abs, absorption_data[:, idx_300K], label='300 K')
ax3.plot(wavelengths_abs, absorption_data[:, idx_420K], label='420 K')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Absorption Cross-section (m²)')
ax3.set_title('Absorption Cross-section at Selected Temperatures')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 1D Plots - Emission for 77K, 300K, 420K
ax4.plot(wavelengths_em, emission_data[:, idx_77K], label='77 K')
ax4.plot(wavelengths_em, emission_data[:, idx_300K], label='300 K')
ax4.plot(wavelengths_em, emission_data[:, idx_420K], label='420 K')
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Emission Cross-section (m²)')
ax4.set_title('Emission Cross-section at Selected Temperatures')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the 1D plot figure
plt.savefig('yb_cross_sections_1d_plots.png', dpi=300, bbox_inches='tight')
print("1D plots saved as 'yb_cross_sections_1d_plots.png'")

# Show both plots
plt.show()
