# EDF - Erbium-Doped Fiber Amplifier Package

A comprehensive Python package for modeling erbium-doped fiber amplifiers (EDFAs) integrated with PyNLO's nonlinear optics simulation framework. This package implements a sophisticated 5-level rate equation model for erbium ions coupled with the nonlinear Schrödinger equation for realistic EDFA simulation.

## Overview

The EDF package provides:
- **5-level erbium ion model** with excited state absorption (ESA)
- **Bidirectional amplification** with forward and backward pumping
- **Coupled rate equations** and nonlinear pulse propagation
- **Adaptive step-size control** for numerical stability
- **Complete integration** with PyNLO's pulse and fiber simulation framework

## Package Structure

```
edf/
├── README.md                      # This file
├── __init__.py                    # Package initialization
├── edfa.py                        # High-level amplification functions
├── edfa_wsplice.py               # EDFA with splice loss modeling
├── five_level_ss_eqns.py         # 5-level rate equation mathematics
├── re_nlse_joint_5level.py       # Main EDF class and models
├── re_nlse_joint_5level_wsplice.py # EDF with splice losses
├── notebooks/
│   └── RE.nb                     # Mathematica derivation of rate equations
└── utility/
    ├── __init__.py              # Utility functions and data
    └── NLight_provided/         # Erbium cross-section and dispersion data
        ├── Erbium Cross Section - nlight_pump+signal.xlsx
        ├── nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx
        ├── nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx
        └── nLIGHT_Er80-8_125-PM_simulated_GVD_dispersion.xlsx
```

## Mathematical Model

### 5-Level Erbium Ion System

The package implements the erbium energy level structure following [Barmenkov et al. JAP 106, 083108 (2009)]:

```
Level 5 (⁴F₉/₂)  ←─── ESA pump (980 nm)
    │ τ₅₄ = 1 μs
Level 4 (⁴I₉/₂)  ←─── ESA signal
    │ τ₄₃ = 5 ns  
Level 3 (⁴I₁₁/₂) ←─── Pump absorption (980 nm)
    │ τ₃₂ = 5.2 μs
Level 2 (⁴I₁₃/₂) ←─── Signal absorption/emission (1530-1600 nm)
    │ τ₂₁ = 10 ms
Level 1 (⁴I₁₅/₂) ←─── Ground state
```

### Key Parameters

- **Cross-section ratios:**
  - `ξₚ = 1.08`: Pump emission/absorption ratio (σ₃₁/σ₁₃)
  - `εₚ = 0.95`: Pump ESA/absorption ratio (σ₃₅/σ₁₃)  
  - `εₛ = 0.17`: Signal ESA/absorption ratio (σ₂₄/σ₁₂)

- **Lifetimes:**
  - `τ₂₁ = 10 ms`: Metastable state (main gain transition)
  - `τ₃₂ = 5.2 μs`: Intermediate state
  - `τ₄₃ = 5 ns`: Higher excited state
  - `τ₅₄ = 1 μs`: Highest excited state

### Steady-State Population Solutions

The population densities (n₁, n₂, n₃, n₄, n₅) are solved analytically from the steady-state rate equations. The solutions are complex expressions derived in Mathematica (see `notebooks/RE.nb`) and implemented in `five_level_ss_eqns.py`.

## Core Classes

### EDF Class

The main class inheriting from `pynlo.materials.SilicaFiber`:

```python
from edf.re_nlse_joint_5level import EDF

edf = EDF(
    f_r=100e6,           # Repetition frequency (Hz)
    overlap_p=1.0,       # Pump mode overlap
    overlap_s=1.0,       # Signal mode overlap  
    n_ion=7e24,          # Ion concentration (ions/m³)
    a_eff=3.14e-12,      # Effective area (m²)
    sigma_p=sigma_p,     # Pump cross-section at 980 nm
    sigma_a=sigma_a,     # Signal absorption cross-sections
    sigma_e=sigma_e      # Signal emission cross-sections
)
```

### Mode Class

Custom mode class with EDF-specific functionality:

- **Dynamic gain calculation** based on population inversions
- **Pump power evolution** via RK45 integration
- **Population density tracking** (n₁-n₅)
- **Bidirectional propagation support**

### Model Classes

- **Model_EDF**: Extends `pynlo.model.Model` with adaptive stepping
- **NLSE**: Standard NLSE propagation with EDF gain

## Usage Examples

### Basic EDFA Simulation

```python
import numpy as np
import pynlo
from edf.re_nlse_joint_5level import EDF
from edf import edfa
from edf.utility import crossSection, ER80_4_125_betas

# Load cross-sections and dispersion data
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e
polyfit_n = ER80_4_125_betas().polyfit

# Create input pulse
pulse = pynlo.light.Pulse.Sech(
    n=256,
    v_min=c/1750e-9,
    v_max=c/1400e-9, 
    v0=c/1560e-9,
    e_p=35e-3/2/f_r,
    t_fwhm=100e-15,
    min_time_window=10e-12
)

# Configure EDF
sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c/980e-9)

edf = EDF(
    f_r=200e6,
    n_ion=80/10*np.log(10)/spl_sigma_a(c/1530e-9),
    a_eff=np.pi*(3.06e-6/2)**2,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e
)
edf.set_beta_from_beta_n(pulse.v0, polyfit_n)

# Generate model and simulate
model = edf.generate_model(pulse, Pp_fwd=2.0)
sim = model.simulate(length=1.5, n_records=100)

# Access results
pump_power = sim.Pp
signal_power = sim.p_v
populations = [sim.n1_n, sim.n2_n, sim.n3_n, sim.n4_n, sim.n5_n]
```

### Bidirectional Amplification

For complete bidirectional simulation with iterative convergence:

```python
from edf import edfa

model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=input_pulse,
    p_bck=None,              # Or provide backward seed
    edf=edf,
    length=1.5,
    Pp_fwd=2.0,             # Forward pump power (W)
    Pp_bck=2.0,             # Backward pump power (W)
    n_records=100,
    tolerance=1e-3          # Convergence tolerance
)
```

### Rate Equations Only

For validation or when nonlinear effects are negligible:

```python
from edf.five_level_ss_eqns import (
    dPp_dz, dPs_dz, gain,
    n1_func, n2_func, n3_func, n4_func, n5_func
)

# Calculate population densities
n1 = n1_func(n_ion, a_eff, overlap_p, overlap_s, 
             nu_p, P_p, nu_s, P_s, sigma_p, sigma_a, sigma_e,
             eps_p, xi_p, eps_s, tau_21, tau_32, tau_43, tau_54)

# Calculate gain coefficient
g = gain(n_ion, a_eff, overlap_p, overlap_s,
         nu_p, P_p, nu_s, P_s, sigma_p, sigma_a, sigma_e,
         eps_p, xi_p, eps_s, tau_21, tau_32, tau_43, tau_54)
```

## Advanced Features

### Splice Loss Modeling

The `edfa_wsplice` module supports **one splice position** that divides the fiber into two segments with different properties:

```python
from edf.re_nlse_joint_5level_wsplice import EDF
from edf import edfa_wsplice

# Create EDF with splice at specific position
edf = EDF(
    # ... fiber parameters for segment 1 ...
    n_ion_1=n_ion_1,           # Ion concentration in first segment
    n_ion_2=n_ion_2,           # Ion concentration in second segment
    a_eff_1=a_eff_1,           # Effective area in first segment
    a_eff_2=a_eff_2,           # Effective area in second segment
    z_spl=0.75,                # Splice position (m)
    loss_spl=0.95,             # Splice transmission (linear, not dB)
    # ... other parameters ...
)

# Simulate with splice loss
model_fwd, sim_fwd, model_bck, sim_bck = edfa_wsplice.amplify(
    p_fwd=input_pulse,
    p_bck=None,
    beta_1=beta_1,             # Dispersion coefficients for segment 1
    beta_2=beta_2,             # Dispersion coefficients for segment 2
    edf=edf,
    length=1.5,
    Pp_fwd=2.0,
    Pp_bck=2.0
)
```

### Shock Wave Effects

```python
# Automatic shock time calculation
model = edf.generate_model(pulse, t_shock="auto", raman_on=True)

# Manual shock time specification  
model = edf.generate_model(pulse, t_shock=1e-12, raman_on=True)
```

### Custom Cross-Sections

```python
# Use custom absorption/emission spectra
edf = EDF(
    # ... other parameters ...
    sigma_a=custom_absorption_spectrum,
    sigma_e=custom_emission_spectrum
)
```

## Data Sources

The package includes experimentally measured data from nLIGHT:

- **Cross-sections**: Absorption and emission spectra for Er³⁺ ions
- **Fiber dispersion**: Group velocity dispersion for Er80 and Er110 fibers
- **Effective areas**: Mode field calculations for different fiber designs

## Integration with PyNLO

The EDF package seamlessly integrates with PyNLO's ecosystem:

- **Pulse objects**: Compatible with all PyNLO pulse types
- **Fiber materials**: Inherits from `SilicaFiber` class
- **Propagation models**: Uses PyNLO's adaptive step-size algorithms
- **Visualization**: Compatible with PyNLO's plotting functions

## Dependencies

- **numpy**: Array operations and mathematical functions
- **scipy**: Scientific computing (constants, interpolation, integration)
- **pynlo**: Core nonlinear optics framework
- **pandas**: Data file reading
- **matplotlib**: Plotting (for examples)

## References

1. Barmenkov et al., "Gain characteristics of a heavily ytterbium-erbium-codoped fiber laser amplifier," *J. Appl. Phys.* **106**, 083108 (2009)
2. nLIGHT Corporation fiber specifications and cross-section data
3. PyNLO documentation: [PyNLO repository](https://github.com/peter-chang62/PyNLO)

## Examples and Validation

See the `examples/edfa_examples/` directory for comprehensive examples:

- `simple_edfa.py`: Basic amplification simulation
- `re_only_5level.py`: Rate equations validation
- `simple_edfa_with_splice.py`: Splice loss modeling
- Various laser configurations (NALM, PBS-based, etc.)

## License

This package is distributed under the same license as PyNLO. See the main repository for license details.