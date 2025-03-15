from ctypes import POINTER, Structure, c_int32, c_void_p,  c_float
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch


class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)


def clip(x, min_value, max_value):
    return torch.clamp(x, min_value, max_value)


def test(
    lib,
    handle,
    torch_device,
    a_shape,
    min_value,
    max_value,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing Clip on {torch_device} with a_shape:{a_shape} min_value:{min_value} max_value:{max_value} dtype:{tensor_dtype}"
    )

    a = torch.randn(a_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.empty(a_shape, dtype=tensor_dtype).to(torch_device)
    min_value = min_value if min_value is not None else torch.finfo(tensor_dtype).min
    max_value = max_value if max_value is not None else torch.finfo(tensor_dtype).max
    ans = clip(a, min_value, max_value)
    
    a_tensor = to_tensor(a, lib)
    c_tensor = to_tensor(c, lib)
    descriptor = infiniopClipDescriptor_t()
    min_value_t = c_float(min_value)
    max_value_t = c_float(max_value)
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            ctypes.byref(min_value_t) if min_value else None,
            ctypes.byref(max_value_t) if max_value else None,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopClip(descriptor, c_tensor.data, a_tensor.data, None)
    )
    assert torch.allclose(c, ans, atol=1e-3, rtol=0)
   #比较 c 和 ans 是否完全相同
    # if torch.equal(c, ans):
    #     print("The tensors are exactly the same.")
    # else:
        
    #     print("The tensors are different.")
    #     print("init tensor a =")
    #     print(a)
    #     print("operators clip c =")
    #     print(c)
    #     print("torch clip ans =")
    #     print(ans)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for a_shape, min_value, max_value in test_cases:
        test(lib, handle, "cpu", a_shape, min_value, max_value, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", a_shape, min_value, max_value, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for a_shape, min_value, max_value in test_cases:
        test(lib, handle, "cuda", a_shape, min_value, max_value, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", a_shape, min_value, max_value, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)




if __name__ == "__main__":
    test_cases = [
        # a_shape, min_value, max_value
        ((1, 3), -0.5, 0.5),
        ((3, 3), -1.0, 1.0),
        ((2, 20, 3), -0.1, 0.1),
        ((32, 20, 512), -0.2, 0.2),
        ((32, 256), -0.3, 0.3),
        ((2, 4, 3), -0.4, 0.4),
        ((2, 3, 4, 5), -0.5, 0.5),
        ((3, 2, 4, 5), -0.6, 0.6),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]
    #test_cpu(lib, test_cases)
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.musa):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")