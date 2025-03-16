#include "../utils.h"
#include "operators.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/gather_cuda.cuh"
#endif

infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,
                                              infiniopGatherDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t input,
                                              infiniopTensorDescriptor_t indices,
                                              infiniopTensorDescriptor_t output,
                                              int64_t axis) {
    if (handle->device == DevCpu) {
#ifdef ENABLE_CPU
        return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *)desc_ptr,
                                       input, indices, output, axis);
#endif
    } else if (handle->device == DevNvGpu) {
#ifdef ENABLE_NV_GPU
        return cudaCreateGatherDescriptor((CudaHandle_t)handle,
                                        (GatherCudaDescriptor_t *)desc_ptr,
                                        input, indices, output, axis);
#endif
    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
                               void *output,
                               void const *input,
                               void const *indices,
                               void *stream) {
    if (desc->device == DevCpu) {
#ifdef ENABLE_CPU
        return cpuGather((GatherCpuDescriptor_t)desc, output, input, indices);
#endif
    } else if (desc->device == DevNvGpu) {
#ifdef ENABLE_NV_GPU
        return cudaGather((GatherCudaDescriptor_t)desc, output, input, indices, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc) {
    if (desc->device == DevCpu) {
#ifdef ENABLE_CPU
        return cpuDestroyGatherDescriptor((GatherCpuDescriptor_t)desc);
#endif
    } else if (desc->device == DevNvGpu) {
#ifdef ENABLE_NV_GPU
        return cudaDestroyGatherDescriptor((GatherCudaDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}