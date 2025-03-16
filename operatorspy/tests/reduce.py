from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time
import torch
from typing import List, Tuple

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

# 性能分析配置
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# 定义三个算子的描述符结构
class ReduceMinDescriptor(Structure):
    _fields_ = [("device", c_int32)]

class ReduceMaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]

class ReduceMeanDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopReduceMinDescriptor_t = POINTER(ReduceMinDescriptor)
infiniopReduceMaxDescriptor_t = POINTER(ReduceMaxDescriptor)
infiniopReduceMeanDescriptor_t = POINTER(ReduceMeanDescriptor)

# Torch参考实现
def reduce_min(x, axes, keepdim=False):
    return torch.amin(x, dim=axes, keepdim=keepdim)

def reduce_max(x, axes, keepdim=False):
    return torch.amax(x, dim=axes, keepdim=keepdim)

def reduce_mean(x, axes, keepdim=False):
    return torch.mean(x, dim=axes, keepdim=keepdim)

# 辅助函数
def tuple_to_int_p(py_tuple: Tuple):
    array = ctypes.c_int * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.POINTER(ctypes.c_int))

def test_reduce(
    lib,
    handle,
    torch_device,
    x_shape,
    axes,
    keepdim,
    op_type,
    tensor_dtype=torch.float32,
):
    print(
        f"Testing {op_type} on {torch_device} with x_shape:{x_shape}, "
        f"axes:{axes}, keepdim:{keepdim}, dtype:{tensor_dtype}"
    )
    
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    
    # 选择对应的reduce操作
    if op_type == "min":
        torch_op = reduce_min
        create_desc = lib.infiniopCreateReduceMinDescriptor
        reduce_op = lib.infiniopReduceMin
        destroy_desc = lib.infiniopDestroyReduceMinDescriptor
        desc_t = infiniopReduceMinDescriptor_t
    elif op_type == "max":
        torch_op = reduce_max
        create_desc = lib.infiniopCreateReduceMaxDescriptor
        reduce_op = lib.infiniopReduceMax
        destroy_desc = lib.infiniopDestroyReduceMaxDescriptor
        desc_t = infiniopReduceMaxDescriptor_t
    else:  # mean
        torch_op = reduce_mean
        create_desc = lib.infiniopCreateReduceMeanDescriptor
        reduce_op = lib.infiniopReduceMean
        destroy_desc = lib.infiniopDestroyReduceMeanDescriptor
        desc_t = infiniopReduceMeanDescriptor_t
    
    ans = torch_op(x, axes, keepdim)
    y = torch.zeros_like(ans)

    # Profile PyTorch
    if PROFILE:
        for _ in range(NUM_PRERUN):
            ans = torch_op(x, axes, keepdim)
        start_time = time.time()
        for _ in range(NUM_ITERATIONS):
            _ = torch_op(x, axes, keepdim)
        torch_time = (time.time() - start_time) / NUM_ITERATIONS
        print(f"PyTorch time: {torch_time:8f}")

    # 创建tensor和描述符
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = desc_t()

    check_error(
        create_desc(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_int_p(axes),
            len(axes),
        )
    )

    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    # 获取工作空间大小
    workspace_size = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetReduceWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = torch.zeros(int(workspace_size.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    # Profile operator
    if PROFILE:
        for _ in range(NUM_PRERUN):
            check_error(
                reduce_op(
                    descriptor,
                    workspace_ptr,
                    workspace_size,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        start_time = time.time()
        for _ in range(NUM_ITERATIONS):
            check_error(
                reduce_op(
                    descriptor,
                    workspace_ptr,
                    workspace_size,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        op_time = (time.time() - start_time) / NUM_ITERATIONS
        print(f"Operator time: {op_time:8f}")

    # 验证结果
    rtol = 1e-3 if tensor_dtype == torch.float16 else 1e-5
    assert torch.allclose(y, ans, atol=0, rtol=rtol)
    check_error(destroy_desc(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axes, keepdim, tensor_dtype in test_cases:
        for op_type in ["min", "max", "mean"]:
            test(lib, handle, "cpu", x_shape, axes, keepdim, op_type, torch.float32)
            test(lib, handle, "cpu", x_shape, axes, keepdim, op_type, torch.float16)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, axes, keepdim, tensor_dtype in test_cases:
        for op_type in ["min", "max", "mean"]:
            test(lib, handle, "cuda", x_shape, axes, keepdim, op_type, torch.float32)
            test(lib, handle, "cuda", x_shape, axes, keepdim, op_type, torch.float16)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # x_shape, axes, keepdim, dtype
        ((3, 4, 5), (1,), False),
        ((2, 3, 4, 5), (0, 2), True),
        ((1,), (0,), False),
        ((1, 1, 1), (0, 1), True),
        ((7, 11, 13), (1,), False),
        ((32, 17, 23, 19), (0, 2), True),
        ((128, 256), (1,), False),
        ((64, 128, 256), (0, 2), True),
        ((3, 5, 7, 9, 11), (0, 2, 4), False),
        ((15, 13, 11, 9), (1, 3), True),
        ((5, 7, 9), (1,), False),
        ((8, 1, 16, 1, 32), (1, 3), False),
        ((1, 64, 1, 128), (0, 2), True),
    ]

    args = get_args()
    lib = open_lib()
    
    # 设置库函数接口
    for op in ["Min", "Max", "Mean"]:
        getattr(lib, f"infiniopCreateReduce{op}Descriptor").restype = c_int32
        getattr(lib, f"infiniopCreateReduce{op}Descriptor").argtypes = [
            infiniopHandle_t,
            POINTER(getattr(globals(), f"infiniopReduce{op}Descriptor_t")),
            infiniopTensorDescriptor_t,
            infiniopTensorDescriptor_t,
            POINTER(ctypes.c_int),
            c_uint64,
        ]
        
        getattr(lib, f"infiniopReduce{op}").restype = c_int32
        getattr(lib, f"infiniopReduce{op}").argtypes = [
            getattr(globals(), f"infiniopReduce{op}Descriptor_t"),
            c_void_p,
            c_uint64,
            c_void_p,
            c_void_p,
            c_void_p,
        ]
        
        getattr(lib, f"infiniopDestroyReduce{op}Descriptor").restype = c_int32
        getattr(lib, f"infiniopDestroyReduce{op}Descriptor").argtypes = [
            getattr(globals(), f"infiniopReduce{op}Descriptor_t"),
        ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)
        
    print("\033[92mTest passed!\033[0m")