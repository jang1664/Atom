from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_acim',
    ext_modules=[
        CUDAExtension(
            name='gemm_acim', 
            sources=[
            'gemm_acim_wrap.cpp',
            '../../src/GEMM/gemm_acim.cu'],
            extra_compile_args={'cxx': ['-O3'],
            'nvcc': ["-O3"]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })