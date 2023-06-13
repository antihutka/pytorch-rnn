from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ptrnn_cpp',
      ext_modules=[cpp_extension.CppExtension('ptrnn_cpp', ['ptrnn.cpp'], extra_compile_args={'cxx':['-O3', '-march=native', '-mprefer-vector-width=512', '-ffast-math', '-fopenmp']}, extra_link_args=['-lgomp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
