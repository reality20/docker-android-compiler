from setuptools import setup, Extension
import numpy
import os
import sys

# Define macros for optimization levels
# Users can enable AVX512 by uncommenting the flag or passing it via CFLAGS.
# Current environment defaults to AVX2/FMA which is safe and highly optimized.
# To satisfy the request for "HEAVILY optimize", we use -O3, -ffast-math, and specific architecture flags.

# If the user wants AVX512:
# c_args = ['-O3', '-mavx512f', '-mavx512dq', '-mfma', '-ffast-math', '-fopenmp']

# Default safe optimization (AVX2):
c_args = ['-O3', '-mavx2', '-mfma', '-ffast-math', '-fopenmp']

module = Extension(
    'quantum_backend',
    sources=['engine.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=c_args,
    extra_link_args=['-fopenmp']
)

setup(
    name='quantum_backend',
    version='1.0',
    description='C Optimized Quantum Backend',
    ext_modules=[module],
    script_args=['build_ext', '--inplace']
)
