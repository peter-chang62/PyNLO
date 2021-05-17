# -*- coding: utf-8 -*-

import setuptools

dependencies = [
    ]

setuptools.setup(
    name='PyNLO',
    version='dev',
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    install_requires=dependencies,

    # metadata to display on PyPI
    author='PyNLO authors',
    description='Python nonlinear optics',
    url='https://github.com/pyNLO/PyNLO',
    )

