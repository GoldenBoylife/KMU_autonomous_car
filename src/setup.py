#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [                  # Cython code file with primes() function
         'calibrate_image.pyx'],  # Python code file with primes() function
        annotate=True),                 # enables generation of the html annotation file
)