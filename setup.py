import glob
import os

import torch
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

def build_extensions():
    """Build C++/CUDA extensions based on availability."""
    if True or (torch.cuda.is_available() and CUDA_HOME is not None) or (
        os.getenv("FORCE_CUDA", "0") == "1"
    ):
        device = "cuda"
        pybind_fn = f"pybind_{device}.cu"
    else:
        device = "cpu"
        pybind_fn = f"pybind_{device}.cpp"

    sources = [os.path.join("torchsparse", "backend", pybind_fn)]
    for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
        if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
            fpath.endswith("_cuda.cu") and device == "cuda"
        ):
            sources.append(fpath)

    extension_type = CUDAExtension if device == "cuda" else CppExtension
    extra_compile_args = {
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
        "nvcc": ["-O3", "-std=c++17"],
    }

    return [
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ]

os.environ['MAX_JOBS'] = '32'
setup(
    ext_modules=build_extensions(),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)