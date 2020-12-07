from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ptrnn_cpp',
      ext_modules=[cpp_extension.CppExtension('ptrnn_cpp', ['ptrnn.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
