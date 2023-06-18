# this is from https://gist.github.com/pranavgade20/c629d4134f5cc6998b489892b3f90a1b
from torch import Tensor
import functools
import inspect

import torch
import ctypes


def kernel_function(lib_path=None):
    if inspect.isfunction(lib_path):
        return kernel_function(lib_path=None)(lib_path)
    else:
        if lib_path is None or not isinstance(lib_path, str):
            raise AssertionError("You must specify a library with your kernel")

        def get_arg_type(annotation):
            if annotation == torch.cuda.streams.Stream:
                return ctypes.POINTER(ctypes.c_void_p)
            elif annotation == int:
                return ctypes.c_int
            elif annotation == Tensor:
                return ctypes.c_void_p
            else:
                raise NotImplementedError

        def get_arg_value(annotation, arg):
            if annotation != arg.__class__:
                raise AssertionError(
                    f"signature mismatch: expected {annotation} but passed value of type {arg.__class__}")
            if arg.__class__ == torch.cuda.streams.Stream:
                return ctypes.c_void_p(arg.cuda_stream)
            elif arg.__class__ == int:
                return int(arg)
            elif arg.__class__ == Tensor:
                return arg.data_ptr()
            else:
                raise NotImplementedError

        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                signature = [v.annotation for v in inspect.signature(f).parameters.values()][1:]
                # Load the custom kernel
                lib = ctypes.CDLL(lib_path)

                fn_to_call = f(*([lib] + [None for _ in signature]))
                # Define the argument types and return type for the function
                fn_to_call.argtypes = [get_arg_type(s) for s in signature]
                fn_to_call.restype = None

                stream = torch.cuda.current_stream()

                print("calling our kernel...")
                # Call our CUDA kernel
                ret = fn_to_call(
                    *[get_arg_value(annotation, arg) for annotation, arg in zip(signature, list(args) + [stream])])

                # Synchronize the CUDA stream
                torch.cuda.synchronize()
                return ret

            return wrapper

        return decorator