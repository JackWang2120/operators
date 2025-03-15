from ctypes import POINTER, Structure, c_int32, c_void_p
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


class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopWhereDescriptor_t = POINTER(WhereDescriptor)


def where(condition, a, b):
    return torch.where(condition, a, b)


def test(
    lib,
    handle,
    torch_device,
    condition_shape,
    a_shape,
    b_shape,
    tensor_dtype=torch.float32,
):
    print(
        f"Testing Where on {torch_device} with condition_shape:{condition_shape}, a_shape:{a_shape}, b_shape:{b_shape}, dtype:{tensor_dtype}"
    )

    condition_data = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)
    a_data = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b_data = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    
    ans = where(condition_data.bool(), a_data, b_data)
    c_data = torch.empty_like(ans).to(torch_device)
    # print(a_data)
    # print(b_data)
    # print(condition_data)
    # print(ans)
    condition_tensor = to_tensor(condition_data, lib)
    a_tensor = to_tensor(a_data, lib)
    b_tensor = to_tensor(b_data, lib)
    c_tensor = to_tensor(c_data, lib)
    # print("condition_tensor=",condition_tensor)
    # print("a_tensor=",a_tensor)
    # print("b_tensor=",b_tensor)
    # print("c_tensor=",c_tensor)
    descriptor = infiniopWhereDescriptor_t()

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            condition_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
            c_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    check_error(
        lib.infiniopWhere(
            descriptor,
            c_tensor.data,
            condition_tensor.data,
            a_tensor.data,
            b_tensor.data,
            None,
        )
    )
    # print("c.data=",c_data.data)
    # print("ans=",ans)
    assert torch.allclose(c_data.data, ans, atol=0, rtol=0)
    #比较 c 和 ans 是否完全相同
   
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for condition_shape, a_shape, b_shape in test_cases:
        test(lib, handle, "cpu", condition_shape, a_shape, b_shape, tensor_dtype=torch.float32)
        test(lib, handle, "cpu", condition_shape, a_shape, b_shape, tensor_dtype=torch.float16)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for condition_shape, a_shape, b_shape in test_cases:
        test(lib, handle, "cuda", condition_shape, a_shape, b_shape, tensor_dtype=torch.float32)
        test(lib, handle, "cuda", condition_shape, a_shape, b_shape, tensor_dtype=torch.float16)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # output_shape, a_shape, b_shape
        ((1, 3), (1, 3), (1, 3)),
        ((3, 3), (3, 3), (3, 3)),
        ((2, 20, 3), (2, 20, 3), (2, 20, 3)),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512)),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112)),
        ((2, 4, 3), (2, 4, 3), (2, 4, 3)),
        ((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5)),
        ((3, 2, 4, 5), (3, 2, 4, 5), (3, 2, 4, 5)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopWhereDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")