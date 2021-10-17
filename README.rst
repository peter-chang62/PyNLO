PyNLO: Nonlinear Optics for Python
==================================
This repo is a fork of the original PyNLO. It started as a rewrite of the pulse propagation model and has grown into something that contains many backwards incompatible changes. It is not yet at feature parity with the original, but it still has some things worth checking out!

Complete documentation for the main branch is available on the repo's `GitHub-Page <https://cdfredrick.github.io/PyNLO/build/html/index.html>`_.


Introduction
------------
The PyNLO package provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with materials. It provides many functionalities for representing pulses of light and nonlinear materials. It features **a unified interface** for simultaneously simulating both **three-wave-mixing** processes (such as second order sum and difference frequency generation) and **four-wave-mixing** processes (such as self-phase modulation, and third order sum and difference frequency generation).

In pyNLO, object-oriented programming is used to emulate physical objects. Whenever possible, each physical entity with intrinsic properties – for example an optical pulse or nonlinear fiber – is mapped to a Python class. These classes keep track of the objects’ properties, calculate interactions, and provide simple calculator-type helper functions.

Features:
	- A solver for the propagation of light through optical waveguides with both 2nd and 3rd order nonlinearities. This solver is highly efficient, thanks to an adaptive-step-size implementation of the embedded 4th order Runge-Kutta in-the-interaction-picture (ERK4(3)-IP) method from `Balac and Mahé (2013) <https://doi.org/10.1016/j.cpc.2012.12.020>`_.
	
	- A flexible object-oriented system for treating laser pulses and optical waveguides.
	
	- ...and much more!


Installation
------------
PyNLO requires Python 3. If you do not already have Python, the open-source Anaconda distribution is a good all-in-one collection of Python packages, or if you are looking for a more minimalist installation try Miniconda. In addition to NumPy and SciPy, PyNLO only requires the Numba and mkl_fft packages (and don't forget matplotlib if you want to look at the results).

The easiest way to start using this fork is to download or clone the repo into a directory of your choice, and then add that directory to the top of your `sys.path <https://docs.python.org/3/library/sys.html#sys.path>`_ variable before importing the package.


Contributing
------------
Open a new issue or discussion on the GitHub repository to add suggestions for improvement, questions, or other comments. Since a lot of the code is new, contributing tests and examples is highly appreciated. New additions should be based off of the `develop` branch.


License
-------
PyNLO is licensed under the `GNU LGPLv3 license <https://choosealicense.com/licenses/lgpl-3.0/>`_. This means that you are free to use PyNLO for any project, but all modifications to its source code need to be kept open source. PyNLO is provided "as is" with absolutely no warranty.
