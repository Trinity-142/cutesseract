import os
from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

PACKAGE_NAME = "py_cutesseract"
COMPUTE_CAP = os.environ.get('COMPUTE_CAP', '86')

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "cu_tesseract",
            "tests",
        )
    ),
    ext_modules=[
        CUDAExtension(
            name="py_cutesseract_cuda",
            sources=[
                "cu_tesseract/py_src/lib.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-DTORCH_USE_CUDA_DSA", "-w", '-lineinfo'],
                "nvcc": ["-O3",
                         "-DTORCH_USE_CUDA_DSA",
                         "-w",
                         "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                         "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                         "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                         "--expt-relaxed-constexpr",
                         "--expt-extended-lambda",
                         f"-gencode=arch=compute_{COMPUTE_CAP},code=\"sm_{COMPUTE_CAP}\"",
                         "-Xptxas=-v"
                         ]
            },
            include_dirs=[
                f"{os.path.dirname(os.path.abspath(__file__))}/cu_tesseract/include"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.11",
    # install_requires=["torch<=2.10.0,>=2.8.0"],
    # dependency_links=[
    #     'https://download.pytorch.org/whl/cu129'
    # ]
)
