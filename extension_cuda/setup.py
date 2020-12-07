from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name='ptrnn_cuda',
  ext_modules=[
    CUDAExtension('ptrnn_cuda', ['ptrnn_cuda.cpp', 'ptrnn_cuda_kernel.cu']),
  ],
  cmdclass={'build_ext': BuildExtension})
