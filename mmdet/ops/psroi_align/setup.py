from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='psroi_align_cuda',
    ext_modules=[
        CUDAExtension('psroi_align_cuda', [
            'src/psroi_align_cuda.cpp',
            'src/psroi_align_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
