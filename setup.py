from setuptools import setup, Extension
import os, numpy
from codecs import open
from os import path
from Cython.Build import cythonize

orbits = [Extension(
    'trm.orbits.orbits',
    [os.path.join('trm','orbits','orbits.pyx')],
#    libraries=["m"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fno-strict-aliasing"],
    define_macros   = [('MAJOR_VERSION', '0'),
                       ('MINOR_VERSION', '1')],)]

setup(
    name='trm.orbits',
    version='1',
    description="Elliptical orbits module",
    long_description="""
orbits allows you to calculate elliptical orbits.
""",
    author='Tom Marsh',
    author_email='t.r.marsh@warwick.ac.uk',
    install_requires=['numpy',],

    packages = ['trm', 'trm.orbits'],
    ext_modules=cythonize(orbits),

    scripts=['scripts/smallecc.py' ,'scripts/fittimes.py', 'scripts/fitrvs.py',
             'scripts/ffittimes.py', 'scripts/acctest.py', 'scripts/varplot.py',
             'scripts/stability.py','scripts/extbest.py', 'scripts/cchisq.py',
             'scripts/ptmodel.py','scripts/osculate.py', 'scripts/gentim.py',
             'scripts/checkdev.py','scripts/orbanal.py', 'scripts/hill.py',
             'scripts/tdiff.py'],

      )

