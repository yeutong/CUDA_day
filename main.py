import os
import torch as t
import ctypes
import subprocess

from torch import Tensor
from utils.kernel_func_decorator import kernel_function

kernel_files = "add3_kernel.cu add_kernel.cu"

# Compile the CUDA kernel using nvcc
# compiles the source code in kernel_file into a shared library named kernel.so that can be loaded and run by your Python program.
nvcc_command = "nvcc -shared --compiler-options '-fPIC' -o kernel.so " + kernel_files + " -I /usr/include/torch/csrc/api/include/"
result = subprocess.run(nvcc_command, shell=True, text=True, capture_output=True)

if result.returncode != 0:
    print(f"Error: nvcc compilation failed")
    print(result.stderr.strip())
    exit(1)
else:
    print(f"compilation successful")


# Initialize CUDA variables
device = t.device("cuda")
size = 1024

# Allocate and assign input tensors in GPU memory
a = t.ones(size, device=device)
b = t.ones(size, device=device)
c = t.ones(size, device=device)
d = t.zeros(size, device=device)


@kernel_function(lib_path=f"{os.getcwd()}/kernel.so")
def sum_cuda(lib, a: Tensor, b: Tensor, c: Tensor, size: int, stream: t.cuda.streams.Stream) -> None:
    return lib.add_cuda

sum_cuda(a, b, c, size)
print(c)


@kernel_function(lib_path=f"{os.getcwd()}/kernel.so")
def sum3_cuda(lib, a: Tensor, b: Tensor, c: Tensor, d: Tensor, size: int, stream: t.cuda.streams.Stream) -> None:
    return lib.add3_cuda

sum3_cuda(a, b, c, d, size)
print(d)