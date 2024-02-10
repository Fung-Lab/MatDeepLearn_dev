#!/usr/bin/env python
"""Setup script for the theta package
"""

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        "series",
        sources=["series.pyx", "bessel.c"],
        include_dirs=[numpy.get_include(), "gsl/include"],
        library_dirs=["gsl/lib"],
        libraries=["gsl", "gslcblas"],
        extra_compile_args=["-I./gsl/include"],
        extra_link_args=["-L./gsl/lib"],
    )
]

setup(
    name="inf_functions",
    author="kruskallin",
    author_email="kruskallin@tamu.edu",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy >= 1.13",
    ],
    zip_safe=False,
)
