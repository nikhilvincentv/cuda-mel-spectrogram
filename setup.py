from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_mel",
    version="0.1.0",
    description="Fused STFT + mel filterbank CUDA kernel",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=[
        CUDAExtension(
            name="cuda_mel_ext",
            sources=[
                "src/bindings.cpp",
                "src/kernels.cu",
            ],
            libraries=["cufft", "cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
