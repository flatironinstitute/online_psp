#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:15:44 2018

@author: agiovann
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("online_psp/coord_update.pyx"),
)
