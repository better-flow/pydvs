#!/usr/bin/python3

from distutils.core import setup, Extension
import numpy

# define the extension module
libdvs = Extension('cpydvs', sources=['pydvs.c'],
                   include_dirs=[numpy.get_include()])

# run the setup
setup(name='pydvs',
      version='1.0',
      description='Python toolkit to work with event-based sensors',
      author='Anton Mitrokhin',
      author_email='amitrokh@umd.edu',
      url='https://github.com/ncos/pydvs',
      py_modules=['pydvs'],
      ext_modules=[libdvs]
     )
