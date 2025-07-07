# PyNLO - Python Nonlinear Optics

PyNLO is a Python library for simulating nonlinear light propagation in optical fibers and bulk media. It provides a comprehensive framework for modeling pulse propagation using split-step methods with adaptive step sizing.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Working with Modes](#working-with-modes)
4. [Utility Functions](#utility-functions)
5. [Advanced Features](#advanced-features)
6. [Complete Examples](#complete-examples)
7. [API Reference](#api-reference)
8. [Technical Details](#technical-details)

## Quick Start

Get started with PyNLO in 4 simple steps:

```python
import pynlo

# 1. Create pulse
pulse = pynlo.light.Pulse.Sech(
    n=2**13, v_min=150e12, v_max=250e12, 
    v0=193e12, e_p=1e-12, t_fwhm=50e-15, min_time_window=10e-12
)

# 2. Define mode properties directly
mode = pynlo.media.Mode(
    v_grid=pulse.v_grid,
    beta=pulse.v_grid * 2.2 * 2 * np.pi / 3e8,  # Simple linear dispersion
    g3=1e-3 / (3/2 * 2 * np.pi * pulse.v_grid)  # Kerr nonlinearity
)

# 3. Create propagation model
model = pynlo.model.NLSE(pulse, mode)

# 4. Run simulation
sim = model.simulate(z_grid=1e-3, dz=1e-6, local_error=1e-6, n_records=100)

# View results
sim.plot("frq")  # Frequency domain evolution
sim.plot("time") # Time domain evolution

# Access final pulse properties
print(f"Final pulse energy: {sim.pulse_out.e_p:.2e} J")
```

**Installation Test:** Run the example above to verify your PyNLO installation works correctly.

## Core Concepts

PyNLO is built around three main classes that work together:

### Pulse Class
Represents optical pulses in time and frequency domains with complementary grids.

**Creation Methods:**
- `Pulse.Sech()` - Hyperbolic secant pulse (most common)
- `Pulse.Gaussian()` - Gaussian pulse shape
- `Pulse.CW()` - Continuous wave
- `Pulse.FromPowerSpectrum()` - From measured spectrum

**Key Properties:**
```python
pulse.a_v        # Root-power spectrum (complex)
pulse.p_v        # Power spectrum
pulse.a_t        # Complex temporal envelope
pulse.p_t        # Temporal power
pulse.e_p        # Pulse energy
pulse.v_width().fwhm  # Spectral width (FWHM)
pulse.t_width().fwhm  # Temporal width (FWHM)
# Note: Width methods return objects with .fwhm, .rms, .eqv attributes
```

### Mode Class
Defines optical medium properties including dispersion, loss/gain, and nonlinearities.

**Required Parameters:**
- `v_grid` - Frequency grid [Hz]
- `beta` - Phase coefficient β(ω) [rad/m]

**Optional Parameters:**
- `alpha` - Gain coefficient α(ω) [1/m]
- `g2` - 2nd-order nonlinearity [m/V] (for χ² effects)
- `g2_inv` - Domain inversion boundaries [m] (for quasi-phase matching)
- `g3` - 3rd-order nonlinearity [m²/V²] (for χ³ effects)
- `r3` - Raman response function

**Computed Properties:**
```python
mode.beta2[center_idx]  # GVD coefficient [s²/m] (array)
mode.D[center_idx]      # Dispersion parameter [s/m²] (array)
mode.n[center_idx]      # Effective refractive index (array)
mode.gamma[center_idx]  # Nonlinear coefficient [W⁻¹m⁻¹] (array)
# Note: Mode properties return arrays - use indexing for display
```

### Model Classes
Handle the physics of light propagation.

**NLSE (Nonlinear Schrödinger Equation):**
- For χ³ Kerr and Raman nonlinearities
- Fast and memory efficient
- Standard choice for fiber simulations

**UPE (Unidirectional Propagation Equation):**
- Supports both χ² and χ³ nonlinearities
- Required for frequency conversion (SHG, DFG, etc.)
- More general but computationally intensive

**Usage:**
```python
model = pynlo.model.NLSE(pulse, mode)  # For fiber/χ³ only
model = pynlo.model.UPE(pulse, mode)   # For crystals/χ² + χ³

# Run simulation
sim = model.simulate(
    z_grid=1e-3,      # Propagation distance [m]
    dz=1e-6,          # Initial step size [m]
    local_error=1e-6, # Error tolerance
    n_records=100     # Number of saved points
)
```

## Working with Modes

The Mode class is the most flexible way to define propagation media. Here are common approaches:

### Method 1: Direct Mode Creation (Recommended)

For complete control over the physics:

```python
import numpy as np
import pynlo.utility as util

# Define frequency grid and material properties
v_grid = pulse.v_grid
v0 = 193e12        # Center frequency [Hz]
beta_2 = -22e-27   # GVD [s²/m]
gamma = 1.3e-3     # Nonlinear coefficient [W⁻¹m⁻¹]

# Convert material properties to Mode parameters
omega_0 = 2 * np.pi * v0
beta_func = util.taylor_series(omega_0, [0, 0, beta_2])  # β₀, β₁, β₂
beta = beta_func(v_grid * 2 * np.pi)

# Convert gamma to g3
g3 = util.chi3.gamma_to_g3(v_grid, gamma)

# Create Mode
mode = pynlo.media.Mode(
    v_grid=v_grid,
    beta=beta,
    g3=g3
)
```

### Method 2: Using Pre-made Materials

For convenience with common materials:

```python
# Using built-in material classes
fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(pulse.v0, [beta_2, beta_3])
fiber.gamma = gamma

# Generate model automatically
model = fiber.generate_model(pulse, method="nlse", raman_on=True)
```

### Linear vs Nonlinear Propagation

**Linear Only (dispersion + loss):**
```python
mode = pynlo.media.Mode(v_grid, beta=beta_array, alpha=alpha_array)
```

**χ³ Nonlinear (fibers):**
```python
mode = pynlo.media.Mode(v_grid, beta=beta_array, g3=g3_array, r3=raman_response)
```

**χ² + χ³ Nonlinear (crystals with quasi-phase matching):**
```python
mode = pynlo.media.Mode(v_grid, beta=beta_array, g2=g2_array, g2_inv=z_inv, g3=g3_array)
model = pynlo.model.UPE(pulse, mode)  # UPE required for χ² effects
```

### Quasi-Phase Matching with g2_inv

For practical χ² simulations, quasi-phase matching (QPM) is essential. The `g2_inv` parameter defines domain inversion boundaries for periodically poled materials like PPLN:

```python
import pynlo.utility.chi2 as chi2

# PPLN crystal parameters
length = 10e-3     # 10 mm crystal
period = 19e-6     # 19 μm poling period
d_eff = 27e-12     # pm/V
n_eff = 2.2        # Effective index
a_eff = 50e-12     # m² effective area

# Calculate domain inversions for QPM
dk = 2 * np.pi / period  # Poling wavenumber
z_inv, domains, poled = chi2.domain_inversions(length, dk)

# Generate g2 parameter (arrays required)
chi2_eff = 2 * d_eff   # Convert d_eff to χ²
n_eff_array = np.ones_like(v_grid) * n_eff
a_eff_array = np.ones_like(v_grid) * a_eff
chi2_eff_array = np.ones_like(v_grid) * chi2_eff
g2 = chi2.g2_shg(v0, v_grid, n_eff_array, a_eff_array, chi2_eff_array)

# Create Mode with QPM
mode = pynlo.media.Mode(
    v_grid=v_grid,
    beta=beta,
    g2=g2,
    g2_inv=z_inv,  # Essential for efficient χ² conversion
    g3=g3
)
```

**Key Points:**
- `g2` defines the material's nonlinear strength
- `g2_inv` defines the spatial modulation for phase matching
- Both are required for efficient frequency conversion
- Use `chi2.domain_inversions()` to calculate `g2_inv` from poling parameters

### Z-Dependent Parameters

For tapered fibers or spatially varying properties:

```python
def gamma_taper(z):
    return 1e-3 * (1 + z/1e-3)  # Linear taper

mode = pynlo.media.Mode(v_grid, beta=beta_array, g3=gamma_taper)
```

## Utility Functions

PyNLO provides conversion functions to translate physical material properties into simulation parameters:

### Linear Properties (`chi1` module)

Convert between refractive index, dispersion parameters, and phase coefficients:

```python
import pynlo.utility.chi1 as chi1

# Refractive index ↔ Phase coefficient
beta = chi1.n_to_beta(v_grid, n_array)
n_array = chi1.beta_to_n(v_grid, beta)

# Dispersion parameter D ↔ GVD β₂
beta2 = chi1.D_to_beta2(v_grid, D)
D = chi1.beta2_to_D(v_grid, beta2)
```

### Second-Order Nonlinearity (`chi2` module)

For χ² processes in nonlinear crystals, the key function is `domain_inversions()` which calculates the `g2_inv` parameter essential for quasi-phase matching:

```python
import pynlo.utility.chi2 as chi2

# Step 1: Calculate domain inversions for QPM (most important)
z_length = 10e-3   # Crystal length [m]
period = 19e-6     # Poling period [m]
dk = 2*np.pi / period  # Poling wavenumber
z_inv, domains, poled = chi2.domain_inversions(z_length, dk)

# Step 2: Generate g2 from material properties
d_eff = 27e-12     # pm/V (PPLN d_eff)
chi2_eff = 2 * d_eff
# Convert scalars to arrays as required by chi2 functions
n_eff_array = np.ones_like(v_grid) * n_eff
a_eff_array = np.ones_like(v_grid) * a_eff
chi2_eff_array = np.ones_like(v_grid) * chi2_eff
g2 = chi2.g2_shg(v0, v_grid, n_eff_array, a_eff_array, chi2_eff_array)  # Second harmonic
g2 = chi2.g2_sfg(v0, v_grid, n_eff_array, a_eff_array, chi2_eff_array)  # Sum frequency

# Step 3: Create Mode with both g2 and g2_inv
mode = pynlo.media.Mode(
    v_grid=v_grid, 
    beta=beta, 
    g2=g2,        # Material nonlinearity
    g2_inv=z_inv  # Domain structure for phase matching
)

# IMPORTANT: g2_inv is essential for efficient χ² conversion
# Without proper QPM, conversion efficiency drops dramatically
```

**Domain Inversions Explained:**
- `z_inv` - Array of positions where domains flip sign
- `domains` - Length of each poled region
- `poled` - Binary array indicating domain orientation (0 or 1)

### Third-Order Nonlinearity (`chi3` module)

For χ³ processes in fibers and bulk media:

```python
import pynlo.utility.chi3 as chi3

# Convert between γ and g3
g3 = chi3.gamma_to_g3(v_grid, gamma)
gamma = chi3.g3_to_gamma(v_grid, g3)

# Convert from n₂ Kerr coefficient
gamma = chi3.n2_to_gamma(v_grid, a_eff, n2)
n2 = chi3.gamma_to_n2(v_grid, a_eff, gamma)

# Generate Raman response (silica fiber parameters)
n_points = 2**13
dt = 10e-15
r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15]
b_weights = [0.245*0.21, 96e-15]
rv_grid, r3 = chi3.raman(n_points, dt, r_weights, b_weights)
```

### Creating Beta Arrays

Use Taylor series expansion from dispersion coefficients:

```python
# Material parameters
v0 = 193e12       # Center frequency
beta_2 = -22e-27  # GVD [s²/m]
beta_3 = 60e-42   # TOD [s³/m]

# Create beta function
omega_0 = 2 * np.pi * v0
beta_coeffs = [beta_0, beta_1, beta_2, beta_3]
beta_func = pynlo.utility.taylor_series(omega_0, beta_coeffs)
beta = beta_func(v_grid * 2 * np.pi)
```

## Advanced Features

### Adaptive Step Size Control

PyNLO uses the ERK4(3)-IP method for automatic step size adjustment:

```python
sim = model.simulate(
    length=1e-3,
    dz=1e-6,          # Initial step size
    local_error=1e-6, # Error tolerance (smaller = more accurate)
    n_records=100     # Output sampling points
)
```

**Performance Tips:**
- Larger `local_error` = faster simulation, less accuracy
- Smaller `dz` initial step = more conservative stepping
- More `n_records` = better temporal resolution of results

### Real-time Monitoring

Monitor simulation progress with live plotting:

```python
sim = model.simulate(
    z_grid=1e-3,
    plot=["frq", "time"],  # Plot frequency and time evolution
    alpha=0.2              # Transparency for real-time updates
)
```

### Memory and Performance

**For large simulations:**
- Use `n_records` to control memory usage
- Consider single-precision if memory is limited
- Enable MKL FFT for better performance (install `mkl_fft`)

**FFT Backend Detection:**
PyNLO automatically uses Intel MKL if available:
- MKL found: "USING MKL FOR FFT'S IN PYNLO"
- MKL not found: "NOT USING MKL FOR FFT'S IN PYNLO"

## Complete Examples

### Example 1: Soliton Propagation

```python
import numpy as np
import pynlo

# Soliton parameters
N = 3              # Soliton number
T0 = 50e-15        # Pulse duration
gamma = 1e-3       # Nonlinear coefficient
beta2 = -22e-27    # Anomalous dispersion

# Calculate soliton energy
e_p = N**2 / (gamma / np.abs(beta2) * T0 / 2)

# Create pulse
pulse = pynlo.light.Pulse.Sech(
    n=2**12, v_min=100e12, v_max=500e12,
    v0=193e12, e_p=e_p, t_fwhm=2*np.log(1+np.sqrt(2))*T0, min_time_window=50e-12
)

# Create mode with dispersion and nonlinearity
v_grid = pulse.v_grid
omega_0 = 2 * np.pi * pulse.v0
beta_func = pynlo.utility.taylor_series(omega_0, [0, 0, beta2])
beta = beta_func(v_grid * 2 * np.pi)
g3 = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma)

mode = pynlo.media.Mode(v_grid=v_grid, beta=beta, g3=g3)

# Simulate soliton propagation
model = pynlo.model.NLSE(pulse, mode)
sim = model.simulate(z_grid=5e-3, dz=1e-6, local_error=1e-6, n_records=100)
```

### Example 2: Second Harmonic Generation in PPLN

```python
import numpy as np
import pynlo
import pynlo.utility.chi2 as chi2

# Fundamental pulse at 1550 nm
pulse = pynlo.light.Pulse.Sech(
    n=2**13, v_min=150e12, v_max=250e12,
    v0=193e12, e_p=1e-12, t_fwhm=100e-15, min_time_window=5e-12
)

# PPLN crystal parameters
length = 20e-3     # 20 mm crystal
period = 19.5e-6   # 19.5 μm poling period (optimized for 1550→775 nm)
d_eff = 27e-12     # pm/V
n_eff = 2.2        # Effective index
a_eff = 80e-12     # m² effective area

# Calculate quasi-phase matching
dk = 2 * np.pi / period
z_inv, domains, poled = chi2.domain_inversions(length, dk)

# Generate dispersion (simple approximation)
v_grid = pulse.v_grid
omega_0 = 2 * np.pi * pulse.v0
beta2 = 5e-27      # Normal dispersion in PPLN
beta_func = pynlo.utility.taylor_series(omega_0, [0, 0, beta2])
beta = beta_func(v_grid * 2 * np.pi)

# Calculate g2 for SHG (arrays required)
chi2_eff = 2 * d_eff
n_eff_array = np.ones_like(v_grid) * n_eff
a_eff_array = np.ones_like(v_grid) * a_eff
chi2_eff_array = np.ones_like(v_grid) * chi2_eff
g2 = chi2.g2_shg(pulse.v0, v_grid, n_eff_array, a_eff_array, chi2_eff_array)

# Create Mode with χ² and QPM
mode = pynlo.media.Mode(
    v_grid=v_grid,
    beta=beta,
    g2=g2,        # Material nonlinearity
    g2_inv=z_inv  # QPM structure - essential for SHG efficiency!
)

# Simulate SHG with UPE (required for χ² processes)
model = pynlo.model.UPE(pulse, mode)
sim = model.simulate(z_grid=length, dz=1e-5, local_error=1e-6, n_records=100)

# Check conversion efficiency
fundamental_power = np.sum(sim.pulse_out.p_v[pulse.v_grid < 220e12]) * pulse.dv
second_harmonic_power = np.sum(sim.pulse_out.p_v[pulse.v_grid > 350e12]) * pulse.dv
efficiency = second_harmonic_power / (fundamental_power + second_harmonic_power)
print(f"SHG conversion efficiency: {efficiency:.1%}")
```

### Example 3: Supercontinuum Generation

```python
import pynlo

# High-power short pulse
pulse = pynlo.light.Pulse.Sech(
    n=2**14, v_min=50e12, v_max=750e12,
    v0=193e12, e_p=10e-12, t_fwhm=30e-15, min_time_window=10e-12
)

# Photonic crystal fiber (high nonlinearity, low dispersion)
beta2 = -2e-27
gamma = 10e-3
v_grid = pulse.v_grid

# Include Raman response for realistic supercontinuum
# CRITICAL: Different grid requirements for NLSE vs UPE models
# For NLSE model (recommended for χ³ only):
rtf_grids = pulse.rtf_grids(alias=0)
n_points = rtf_grids.n      # Use rtf_grids.n for NLSE
dt = rtf_grids.dt           # Use rtf_grids.dt for NLSE

# For UPE model (supports both χ² and χ³):
# n_points = pulse.rn       # Use pulse.rn for UPE
# dt = pulse.rdt            # Use pulse.rdt for UPE

r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15]
b_weights = [0.245*0.21, 96e-15]
rv_grid, r3 = pynlo.utility.chi3.raman(n_points, dt, r_weights, b_weights)

# Create mode with Raman
omega_0 = 2 * np.pi * pulse.v0
beta_func = pynlo.utility.taylor_series(omega_0, [0, 0, beta2])
beta = beta_func(v_grid * 2 * np.pi)
g3 = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma)

mode = pynlo.media.Mode(v_grid=v_grid, beta=beta, g3=g3, rv_grid=rv_grid, r3=r3)

# Simulate supercontinuum generation
model = pynlo.model.NLSE(pulse, mode)
sim = model.simulate(z_grid=1e-2, dz=1e-6, local_error=1e-6, n_records=200)
```

## API Reference

### Pulse Creation
```python
Pulse.Sech(n, v_min, v_max, v0, e_p, t_fwhm, min_time_window)
Pulse.Gaussian(n, v_min, v_max, v0, e_p, t_fwhm, min_time_window)
Pulse.CW(n, v_min, v_max, v0, e_p)
```

### Mode Creation
```python
Mode(v_grid, beta, alpha=None, g2=None, g2_inv=None, g3=None, rv_grid=None, r3=None, z=0.0)
```

**Parameter Requirements:**
- `v_grid` - Frequency array [Hz] (required)
- `beta` - Phase coefficient array β(ω) [rad/m] (required)
- `alpha` - Gain coefficient array α(ω) [1/m] (optional)
- `g2` - 2nd-order nonlinearity array [m/V] (optional, for χ² effects)
- `g2_inv` - Domain inversion positions [m] (optional, essential for QPM)
- `g3` - 3rd-order nonlinearity array [m²/V²] (optional, for χ³ effects)
- `rv_grid` - Raman frequency array [Hz] (optional, must match grid requirements)
- `r3` - Raman response array (optional)

**Grid Compatibility Notes:**
- All arrays except `g2_inv` and Raman parameters must match `v_grid` size
- For Raman: NLSE requires `rv_grid` matching `pulse.rtf_grids(alias=0).v_grid`, UPE uses `pulse.rv_grid`

**Typical Usage for PPLN:**
```python
z_inv, _, _ = chi2.domain_inversions(length, 2*pi/period)
mode = Mode(v_grid, beta, g2=g2_array, g2_inv=z_inv)
```

### Model Simulation
```python
NLSE(pulse, mode)
UPE(pulse, mode)
model.simulate(z_grid, dz=None, local_error=1e-6, n_records=100, plot=None)
```

### Key Utility Functions
```python
# chi1 conversions
chi1.n_to_beta(v_grid, n)
chi1.D_to_beta2(v_grid, D)

# chi2 conversions (QPM essential)
chi2.domain_inversions(z, dk)  # Critical for g2_inv parameter
chi2.g2_shg(v0, v_grid, n_eff_array, a_eff_array, chi2_eff_array)  # Arrays required

# chi3 conversions
chi3.gamma_to_g3(v_grid, gamma, t_shock=None)
chi3.raman(n, dt, r_weights, b_weights=None)

# General utilities
taylor_series(x0, fn)
TFGrid(n, v_ref, dv, alias=1)
```

## Technical Details

### Units
All quantities use SI base units:
- Frequency: Hz
- Time: s
- Energy: J
- Length: m
- Power: W

### Mathematical Framework
- Analytic signal representation (positive frequencies only)
- Complementary time-frequency grids via FFT
- ERK4(3)-IP adaptive stepping algorithm
- Efficient numba JIT compilation for performance

### Dependencies
- **numpy**: Array operations
- **scipy**: Scientific computing
- **numba**: JIT compilation
- **mkl_fft**: Intel MKL FFT (optional, improves performance)
- **matplotlib**: Plotting (required for examples)

## Troubleshooting

### Common Warnings

#### Overflow in Sech Pulse Creation
```
RuntimeWarning: overflow encountered in cosh
```
**Cause**: Large time windows in Sech pulse creation cause numerical overflow in the hyperbolic cosine function  
**Impact**: **Non-critical warning** - does not affect pulse creation or simulation accuracy  
**Solution**: This warning can be safely ignored. It occurs during pulse initialization for large time grids and is expected behavior. The pulse is created correctly despite the warning.

#### FFT Backend Message
```
NOT USING MKL FOR FFT'S IN PYNLO
```
**Cause**: Intel MKL FFT library not available  
**Impact**: Performance only, functionality intact  
**Solution**: Install `mkl_fft` package for better performance (optional)

### Common Errors

#### Grid Size Mismatch for Raman
```
AssertionError: The pulse and mode must be defined over the same frequency grid
```
**Cause**: Incorrect Raman grid parameters for model type  
**Solution**: 
- **NLSE**: Use `pulse.rtf_grids(alias=0).n` and `pulse.rtf_grids(alias=0).dt`
- **UPE**: Use `pulse.rn` and `pulse.rdt`

#### Chi2 Function Array Requirements
```
ValueError: object of too small depth for desired array
```
**Cause**: Chi2 functions expect arrays, not scalars  
**Solution**: Convert scalars to arrays:
```python
n_eff_array = np.ones_like(v_grid) * n_eff
a_eff_array = np.ones_like(v_grid) * a_eff
chi2_eff_array = np.ones_like(v_grid) * chi2_eff
```

#### Parameter Name Errors
```
TypeError: got an unexpected keyword argument
```
**Cause**: Incorrect parameter names  
**Common fixes**:
- `n_points` → `n`
- `time_window` → `min_time_window`  
- `length` → `z_grid`

### Performance Tips

- Use Intel MKL FFT (`pip install mkl_fft`) for better performance
- Larger `local_error` values = faster simulation, less accuracy
- Use appropriate `n_records` to balance memory usage and temporal resolution
- For large simulations, consider using smaller precision if memory is limited

For more examples, see `examples/pynlo_examples/` in the repository.