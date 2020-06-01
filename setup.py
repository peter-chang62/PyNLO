# -*- coding: utf-8 -*-

import setuptools

dependencies = [
    "jsonpickle >= 0.9.2",
    "jsonschema >= 2.5.1",
    "numpy >= 1.9.2",
    "scipy >= 0.15.1",
    ]

setuptools.setup(
    name='pyNLO',
    version='0.1.2',
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    install_requires=dependencies,

    # metadata to display on PyPI
    author='pyNLO authors',
    description='Python nonlinear optics',
    url='https://github.com/pyNLO/PyNLO',
    )

