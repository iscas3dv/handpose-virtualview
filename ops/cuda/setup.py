from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

arch_list = []
for i in range(torch.cuda.device_count()):
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(i))
    if arch not in arch_list:
        arch_list.append(arch)
arch_list = ';'.join(arch_list)
os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

setup(
    name='render_cuda',
    ext_modules=[
        CUDAExtension('depth_to_point_cloud_mask_cuda', [
            'depth_to_point_cloud_mask_cuda.cpp',
            'depth_to_point_cloud_mask_cuda_kernel.cu',
        ]),
        CUDAExtension('point_cloud_mask_to_depth_cuda', [
            'point_cloud_mask_to_depth_cuda.cpp',
            'point_cloud_mask_to_depth_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })