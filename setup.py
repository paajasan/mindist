import sys, os
import numpy as np
from numpy.distutils.core import Extension, setup
import subprocess as subp

def run_command(command,dir):
    prevdir = os.getcwd()

    try:
        os.chdir(dir)
        cp = subp.run(command)
        if(cp.returncode!=0):
            sys.exit(cp.returncode)

    finally:
        os.chdir(prevdir)


mindist_ext = Extension(
                "mindist._f2py.mindist", ["lib/mindist.f90"],
                extra_link_args=['-lgomp'], extra_f90_compile_args=["-fopenmp"],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                )


run_command(command=["gfortran", "-c", "-Wall", "-fPIC",  "-Ofast",
                     "stdlib_sorting_sort_index_real.f90", "mindist_grid_utils.f90",
                     "mindist_PBC_utils.f90", "mindist_PBC_grid_utils.f90"],
            dir="lib")


mindist_ext_grid = Extension(
                "mindist._f2py.mindist_grid", ["lib/mindist_grid.f90"],
                extra_link_args=['-lgomp'], extra_f90_compile_args=["-fopenmp"],
                extra_objects=["lib/mindist_grid_utils.o",
                               "lib/stdlib_sorting_sort_index_real.o"],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                )


mindist_ext_PBC = Extension(
                "mindist._f2py.mindist_PBC", ["lib/mindist_PBC.f90"],
                extra_link_args=['-lgomp'], extra_f90_compile_args=["-fopenmp"],
                extra_objects = ["lib/mindist_PBC_utils.o"],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                )


mindist_ext_PBC_grid = Extension(
                "mindist._f2py.mindist_PBC_grid", ["lib/mindist_PBC_grid.f90"],
                extra_link_args=['-lgomp'], extra_f90_compile_args=["-fopenmp"],
                extra_objects = ["lib/mindist_PBC_utils.o",
                                 "lib/mindist_PBC_grid_utils.o",
                                 "lib/stdlib_sorting_sort_index_real.o"],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                )


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    packages    = ["mindist", "mindist._f2py"],
    package_dir = {"mindist": "src", "mindist._f2py": "lib"},
    ext_modules=[mindist_ext, mindist_ext_grid,
                 mindist_ext_PBC, mindist_ext_PBC_grid]
)