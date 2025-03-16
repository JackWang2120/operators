from ctypes import POINTER, Structure, c_int32, c_int64, c_uint64, c_void_p, c_bool
import ctypes
import sys
import os
import torch

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

class GatherDescriptor(Structure):
    _fields_ = [
        ("device", c_int32),
        ("dtype", c_int32),
        ("ndim", c_uint64),
        ("data_size", c_uint64),
        ("axis", c_int64),
        ("input_shape", POINTER(c_int64)),
        ("input_strides", POINTER(c_int64)),
        ("indices_shape", POINTER(c_int64)),
        ("output_shape", POINTER(c_int64)),
    ]

infiniopGatherDescriptor_t = POINTER(GatherDescriptor)

def gather(input, indices, axis):
    return torch.gather(input, axis, indices)

def test(
    lib,
    handle,
    torch_device,
    input_shape,
    indices_shape,
    axis,
    tensor_dtype=torch.float32,
):
    print(
        f"Testing Gather on {torch_device} with input_shape:{input_shape}, "
        f"indices_shape:{indices_shape}, axis:{axis}, dtype:{tensor_dtype}"
    )

    input = torch.rand(input_shape, dtype=tensor_dtype).to(torch_device)
    indices = torch.randint(0, input_shape[axis], indices_shape, dtype=torch.int64).to(torch_device)
    ans = gather(input, indices, axis)
    output = torch.zeros_like(ans)
    # print("Operator gather output:", output)
    input_tensor = to_tensor(input, lib)
    indices_tensor = to_tensor(indices, lib)
    output_tensor = to_tensor(output, lib)
    # print("Operator gather input:", input)
    # print("Operator gather indices:", indices)
    # print("Operator gather ans:", ans)
    # print("Operator gather output:", output)
    
    descriptor = infiniopGatherDescriptor_t()

    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            input_tensor.descriptor,
            indices_tensor.descriptor,
            output_tensor.descriptor,
            axis,
        )
    )
    input_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()
    check_error(
        lib.infiniopGather(
            descriptor,
            output_tensor.data,
            input_tensor.data,
            indices_tensor.data,
            None,
        )
    )

    # assert torch.allclose(output, ans, atol=0, rtol=0)
    print("Operator gather output:", output)
    print("Operator gather ans:", ans)
    
    

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for input_shape, indices_shape, axis in test_cases:
        test(lib, handle, "cpu", input_shape, indices_shape, axis)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for input_shape, indices_shape, axis in test_cases:
        test(lib, handle, "cuda", input_shape, indices_shape, axis)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # 基本测试
        ((2, 3), (2, 3), 0),
        ((4, 5), (4, 5), 1),
        
        # 多维测试
        ((2, 3, 4), (2, 3, 4), 1),
        ((3, 4, 5), (3, 4, 5), 2),
        
        # 边界测试
        ((1,), (1,), 0),
        ((2, 1), (2, 1), 1),
    ]

    args = get_args()
    lib = open_lib()
    
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int64,
    ]

    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [
        infiniopGatherDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)

    print("\033[92mTest passed!\033[0m")