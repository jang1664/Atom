from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_acim_v1',
    ext_modules=[
        CUDAExtension(
            name='gemm_acim_v1', 
            sources=[
            'gemm_acim_wrap.cpp',
            '../../src/GEMM/gemm_acim_v1.cu'],
            extra_compile_args={'cxx': ['-O3', '-I', '../../include'],
            'nvcc': ["-O3", '-I', '../../include']}
            # extra_compile_args={'cxx': ['-O3', '-I', '../../include', "-D", "DEBUG"],
            # 'nvcc': ["-O3", '-I', '../../include', "-D", "DEBUG"]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
