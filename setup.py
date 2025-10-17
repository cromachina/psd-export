import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy

cflags = []
if sys.platform == 'win32':
    cflags.append('/d2FH4-')
else:
    cflags.extend(['-march=native'])

extensions = [
    Extension(
        '*',
        ['**/*.pyx'],
        extra_compile_args=cflags,
        include_dirs=[numpy.get_include()],
    )
]

setup(ext_modules=cythonize(extensions, annotate=True))
