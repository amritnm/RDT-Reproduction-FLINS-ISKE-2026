"""
Setup script for compiling Cython RDT modules.

Usage:
    python setup.py build_ext --inplace  # Compile in-place
    pip install -e .                      # Install in development mode
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

# Compiler flags for maximum performance
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # Windows (MSVC)
    extra_compile_args = ['/O2', '/fp:fast']
else:
    # Linux/Mac (GCC/Clang)
    extra_compile_args = ['-O3', '-ffast-math', '-march=native']
    # OpenMP support (uncomment if you want parallel tree building)
    # extra_compile_args.append('-fopenmp')
    # extra_link_args.append('-fopenmp')

# Define extensions
extensions = [
    Extension(
        "rdt_cython.src._criterion",
        ["src/_criterion.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        "rdt_cython.src._tree_cython",
        ["src/_tree_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

setup(
    name="rdt_cython",
    version="1.0.0",
    description="Ultra-fast Cython implementation of Restricted Decision Trees",
    author="Your Name",
    packages=["rdt_cython", "rdt_cython.src"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
            'profile': False,
        },
        annotate=True,  # Generate HTML annotation files for optimization analysis
    ),
    install_requires=[
        'numpy>=1.18.0',
        'cython>=0.29.0',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)
