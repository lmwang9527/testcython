from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("testcython", ["testcython.pyx", "randomkit.c"],
    include_dirs = [numpy.get_include()])
                
#mtrand = Extension("mtrand", ["mtrand.pyx"],
#    include_dirs = [numpy.get_include()])

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
