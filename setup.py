from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import Cython

directive_defaults = Cython.Compiler.Options.get_directive_defaults() 
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

include = [np.get_include()]

setup(
    ext_modules = cythonize("*.pyx"),include_dirs=include
    )