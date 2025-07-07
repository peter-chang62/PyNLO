# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- Run tests: `python -m pytest tests/`
- Run specific test file: `python -m pytest tests/test_light.py`
- Run example scripts to verify functionality: `python examples/pynlo_examples/optical-solitons.py`

### Documentation
- Build documentation: `cd docs && make html`
- Documentation is built using Sphinx and stored in `docs/build/`

### Python Environment
- This project uses Python 3 with dependencies: numpy, scipy, numba, mkl_fft (optional)
- Type checking configuration is in `pyrightconfig.json` with relaxed settings for scientific computing
- Virtual environment named "idp" is configured in pyrightconfig.json

## Code Architecture

### Core Package Structure
- **pynlo/**: Main package with core nonlinear optics functionality
  - `light.py`: Pulse class for representing optical pulses in time/frequency domains
  - `model.py`: Propagation models (Model, NLSE, UPE) for simulating light through materials
  - `media.py`: Mode class for optical modes in waveguides and materials
  - `device.py`: Device-level abstractions
  - `materials.py`: Material property definitions
  - `utility/`: Helper functions and utilities

### Key Classes and Concepts
- **Pulse**: Central class for optical pulses with time/frequency domain representations
- **Mode**: Represents optical modes with linear/nonlinear properties (beta, alpha, gamma coefficients)
- **Model/NLSE/UPE**: Propagation models using split-step methods with adaptive step sizing
- **TFGrid**: Time-frequency grid system for analytic signal representation

### Additional Packages
- **edf/**: Erbium-doped fiber amplifier (EDFA) simulation with 5-level rate equations
- **eydf/**: Erbium-Ytterbium co-doped fiber amplifier (EYDFA) with 7-level rate equations
- Both support forward/backward pumping and can be combined with PyNLO for complete laser simulations

### Mathematical Framework
- All quantities in base SI units (Hz, s, J)
- Uses analytic signal representation with positive frequencies only
- Efficient FFT-based algorithms with numba JIT compilation
- Adaptive step-size ERK4(3)-IP method for propagation

### FFT Implementation
- PyNLO automatically detects and uses Intel MKL FFT if available for better performance
- Falls back to numpy/scipy FFT implementations if MKL is not installed
- Import message displays which FFT backend is being used: "USING MKL FOR FFT'S IN PYNLO" or "NOT USING MKL FOR FFT'S IN PYNLO"
- FFT interface in `pynlo/utility/fft.py` provides consistent API regardless of backend

### Utility Functions
- **fft.py**: FFT interface with automatic MKL/numpy backend selection
- **chi1/chi2/chi3.py**: Linear and nonlinear susceptibility calculations
- **clipboard.py**, **blit.py**: Plotting convenience functions for examples
- **TFGrid**: Complementary time-frequency grid system

## Working with the Code

### Running Examples
Examples are in `examples/` directory with subdirectories for different applications:
- `pynlo_examples/`: Basic PyNLO demonstrations
- `edfa_examples/`: EDFA simulations
- `eydfa_examples/`: EYDFA simulations

### Development Guidelines
- Follow existing code style and patterns
- Use numpy/scipy for numerical operations
- Leverage numba JIT compilation for performance-critical code
- Maintain SI unit consistency throughout
- Complex envelope representation assumes analytic signals (positive frequencies)

### Key Dependencies
- numpy: Array operations and mathematical functions
- scipy: Scientific computing utilities
- numba: JIT compilation for performance
- mkl_fft: Intel MKL FFT implementation (optional, improves performance)
- matplotlib: Required for plotting in examples