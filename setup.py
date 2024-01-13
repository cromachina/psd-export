import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "psd_export.rle",
        ["src/psd_export/rle.pyx"],
        extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [],
    )
]

setup(ext_modules=cythonize(extensions))
