User Guide
==========

Time and Frequency Grids
------------------------
The core functionality of this package is built off of the well-defined and internally consistent time and frequency grids as given by :py:class:`pynlo.utility.TFGrid`. These grids represent physical coordinate spaces. Critically, the frequency grid is only defined at positive frequencies and is aligned to the origin. This facilitates Fourier transforms to the time domain with analytic, complex envelope representations as well as with real-valued representations. See the notes on :doc:`Fourier transforms </notes/fourier_transforms>` for more details.


Pulses
------
Optical pulses are represented in PyNLO with :py:class:`pynlo.light.Pulse` objects. These objects contain methods for accessing the time and frequency domain properties of the pulse and of the underlying time and frequency grids. There are several class methods that can conveniently generate :py:class:`~pynlo.light.Pulse` objects, such as with common pulse shapes (:py:meth:`~pynlo.light.Pulse.Sech`, :py:meth:`~pynlo.light.Pulse.Gaussian`, and others) or with existing spectral data (:py:meth:`~pynlo.light.Pulse.FromPowerSpectrum`).


Media
-----
PyNLO represents single optical modes with :py:class:`pynlo.media.Mode` objects. These objects are defined over the frequency domain and contain information due to the coupling of material and spatial properties of a mode. In addition to defining the angular wavenumbers (:math:`\beta`), loss (:math:`\alpha`), and related properties of a mode, :py:class:`~pynlo.media.Mode` objects also contain information about an optical mode's effective nonlinearity. Spatially dependent properties can be input as callable functions in which the first argument is the propagation distance (:math:`z`).


Models
------
The unidirectional propagation models implemented in PyNLO are found in :py:mod:`pynlo.model` and are based on a multi-mode expansion of maxwell's equations. If a mathematically complete set of modes is used in the expansion, the only approximation that occurs is the assumption that the light propagates in a single direction. See the notes on :doc:`nonlinear optics </notes/nonlinear_optics>` for more details. Only single-mode simulations are currently supported, but models that implement the full multi-modal relationships are intended in future updates. Such multi-mode models would not only enable simulations with multi-mode waveguides, but they would also provide a framework for modeling pulse propagation in bulk materials, such as with Hermite or Laguerre free-space Gaussian modes.

The single-mode :py:class:`pynlo.model.SM_UPE` model is initialized using a single :py:class:`~pynlo.light.Pulse` and :py:class:`~pynlo.media.Mode` object. All simulation specific parameters should be entered when running the model's :py:meth:`~pynlo.model.SM_UPE.simulate` method. Real-time visualization of the simulation is available in the frequency, wavelength, or time domains by setting the method's `plot` keyword.

Anti-Aliasing
^^^^^^^^^^^^^
Multiplications of functions in the time domain (operations intrinsic to nonlinear optics) are equivalent to convolutions in the frequency domain and vice versa. The support of a convolution is the sum of the support of its parts. Thus, 2nd and 3rd order processes in the time domain need 2x and 3x number of points in the frequency domain to avoid aliasing.

:py:class:`~pynlo.light.Pulse` objects only initialize the minimum number of points necessary to represent the real-valued time domain pulse (i.e. 1x). While this minimizes the numerical complexity of individual nonlinear operations, aliasing introduces systematic error and can even increase the total simulation time by forcing shorter step sizes. More points can be generated for a specific :py:class:`~pynlo.light.Pulse` object by running its :py:meth:`~pynlo.utility.TFGrid.rtf_grids` method with `update` set to ``True`` and with a `n_harmonic` parameter greater than 1. Anti-aliasing is not always necessary as phase matching can suppress the aliased interactions, but it is best practice to verify that behavior on a case-by-case basis.


Utilities
---------
The :py:mod:`pynlo.utility` module includes helper or calculator type functions in addition to the definition of PyNLO's time and frequency grids. Conversion functions and estimators of physically relevant parameters are organized into modules based on their relationship to the linear (:py:mod:`pynlo.utility.chi1`) or nonlinear (:py:mod:`pynlo.utility.chi2` and :py:mod:`pynlo.utility.chi3`) susceptibilities. The definitions of the fast Fourier transforms used in PyNLO are contained in :py:mod:`pynlo.utility.fft`.


Example Code
------------
.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Examples
   
   examples/silica-pcf_anomalous
   examples/ppln_mismatched-shg
