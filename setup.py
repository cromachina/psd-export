import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy

extensions = [
    Extension(
        "*",
        ["src/psd_export/*.pyx"],
        extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [],
        include_dirs=[numpy.get_include()],
    )
]

setup(ext_modules=cythonize(extensions, annotate=True))
