"""
Setup file for building Cython extensions for Optimal RDT.

To build the extension in-place:
    python setup.py build_ext --inplace

To install:
    pip install -e .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Define Cython extensions
extensions = [
    Extension(
        name="rdt_optimal_cython.src._optimal_tree_cython",
        sources=[os.path.join(here, "src", "_optimal_tree_cython.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "/O2" if os.name == 'nt' else "-O3",  # Windows vs Unix
            "/favor:INTEL64" if os.name == 'nt' else "-march=native",
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# Cythonize with optimal settings
ext_modules = cythonize(
    extensions,
    compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
        'nonecheck': False,
        'embedsignature': True,
    },
    annotate=True,  # Generate HTML annotation files
)

setup(
    name="rdt_optimal_cython",
    version="0.1.0",
    description="Cython-accelerated Optimal Restricted Decision Tree",
    ext_modules=ext_modules,
    packages=['rdt_optimal_cython', 'rdt_optimal_cython.src'],
    package_dir={'rdt_optimal_cython': '.'},
    install_requires=[
        'numpy>=1.19.0',
        'cython>=0.29.0',
    ],
    zip_safe=False,
)
