# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyNLO is a Python package for modeling the nonlinear interaction of light with matter. This is a fork of the original PyNLO with significant rewrites to add 2nd-order nonlinearities to pulse propagation models. The codebase includes:

- **Core PyNLO**: Nonlinear optics simulation (pynlo/ directory)
- **EDF Package**: Erbium-doped fiber amplifier simulation (edf/ directory) 
- **EYDF Package**: Erbium/Ytterbium co-doped fiber amplifier simulation (eydf/ directory)

## Architecture

### Core Components

- **pynlo/light.py**: Defines the `Pulse` class for representing laser pulses in time and frequency domains
- **pynlo/media.py**: Defines the `Mode` class for optical modes in waveguides
- **pynlo/model.py**: Contains simulation models (`Model`, `NLSE`, `UPE`) for light propagation
- **pynlo/device.py**: Device-level abstractions
- **pynlo/materials.py**: Material property definitions
- **pynlo/utility/**: Utility functions including FFT operations, chi1/chi2/chi3 nonlinear optics, and helper functions

### Active Fiber Packages

- **edf/**: Erbium-doped fiber amplifier simulation with 5-level rate equations
- **eydf/**: Erbium/Ytterbium co-doped fiber amplifier with 7-level rate equations

Both packages can simulate forward/backward pumped amplifiers and work with PyNLO for complete laser system modeling.

## Installation and Setup

The package uses setuptools with pyproject.toml configuration. Dependencies are:
- numpy
- scipy  
- mkl_fft
- numba

Install by adding the repository directory to sys.path or symlinking pynlo/ to site-packages.

## Testing

Tests are located in tests/ directory:
- Run tests using standard pytest: `python -m pytest tests/`
- test_light.py: Tests for pulse generation and properties
- test_utility.py: Tests for utility functions

## Documentation

Documentation is built using Sphinx:
- Source files in docs/source/
- Build with: `cd docs && make html`
- API documentation is auto-generated from docstrings

## Development Notes

- All quantities should be in base SI units (Hz for frequency, s for time, J for energy)
- The codebase uses both public methods and private methods (prefixed with `_`)
- Private methods often use FFT-ordered arrays (with ifftshift)
- Public methods maintain monotonically ordered coordinate arrays
- Code uses numba JIT compilation for performance-critical functions

## Example Usage

Extensive examples are provided in examples/ directory:
- examples/pynlo_examples/: Core PyNLO functionality demos
- examples/edfa_examples/: EDFA simulation examples  
- examples/eydfa_examples/: EYDFA simulation examples

Key example files demonstrate:
- Optical soliton propagation
- Supercontinuum generation
- Second-order nonlinear effects (χ² processes)
- Fiber amplifier designs