PyNLO: Nonlinear Optics Modeling for Python
=================================================
This repo is a fork of the original PyNLO. It started as a rewrite of the pulse propagation model and has grown into something that contains many backwards incompatible changes. It is not yet at feature parity with the original, but it still has some things worth checking out!

Complete documentation for the main branch is available on the repo's GitHub-Page.


Introduction
------------
The PyNLO package provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with materials. It provides many functionalities for representing pulses of light and nonlinear materials. Also, it features **a unified interface** for simulating both **three-wave-mixing** processes (such as second order sum and difference frequency generation) and **four-wave-mixing** processes (such as self-phase modulation, and third order sum and difference frequency generation).

In pyNLO, object-oriented programming is used to emulate physical objects. Whenever possible, each physical entity with intrinic properties – for example an optical pulse or nonlinear fiber – is mapped to a Python class. These classes keep track of the objects’ properties, calculate interactions, and provide simple calculator-type helper functions.

Features:
	- A solver for the propagation of light through :math:`\chi^2` and :math:`\chi^3` optical waveguides. This solver is highly efficient, thanks to an adaptive-step-size implementation of the embedded 4th order Runge-Kutta in-the-interaction-picture (ERK4(3)-IP) method from `Balac and Mahé (2013) <https://doi.org/10.1016/j.cpc.2012.12.020>`_.
	
	- A flexible object-oriented system for treating laser pulses and optical waveguides.
	
	- ...and much more!


Installation
------------
PyNLO requires Python 3. If you don't already have Python, the open-source Anaconda distribution is a good all-in-one collection of Python packages, or if you are looking to start out with a more minimalist installation try Miniconda.

The easiest way to start using this fork is to download or clone the repo into a directory of your choice, and then add that directory to the top of your `sys.path <https://docs.python.org/3/library/sys.html#sys.path>`_ variable before importing the package.


Contributing
------------
Open a new issue or discussion on the github repository to add suggestions for improvement, questions, or other comments.


License
-------
PyNLO is licensed under the `GNU LGPLv3 license <https://choosealicense.com/licenses/lgpl-3.0/>`_. This means that you are free to use PyNLO for any project, but all modifications to its source code should be kept open-source. Of course, PyNLO is provided "as is" with absolutely no warrenty.
