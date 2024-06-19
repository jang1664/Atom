from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_acim',
    ext_modules=[
        CUDAExtension('gemm_acim', [
            'gemm_acim_wrap.cpp',
            '../src/GEMM/gemm_acim.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })