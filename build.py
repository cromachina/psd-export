import sys

from Cython.Build import cythonize
import numpy

def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                module_list="**/*.pyx",
                extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [],
                include_path=[numpy.get_include()],
                annotate=True,
            )
        }
    )