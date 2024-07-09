from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemv_acim_v1',
    ext_modules=[
        CUDAExtension(
            name='gemv_acim_v1', 
            sources=[
            'gemm_acim_wrap.cpp',
            '../../src/GEMV/gemv_acim_with_scale_v1.cu'],
            extra_compile_args={'cxx': ['-O3', '-I', '../../include'],
            'nvcc': ["-O3", '-I', '../../include']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
