#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/where_cuda.cuh"
#endif

infiniopStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t condition,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b,
    infiniopTensorDescriptor_t c) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *)desc_ptr, 
                                          condition, a, b, c);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateWhereDescriptor((CudaHandle_t)handle, 
                                           (WhereCudaDescriptor_t *)desc_ptr,
                                           condition, a, b, c);
        }
#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *c,
    void const *condition,
    void const *a,
    void const *b,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuWhere((WhereCpuDescriptor_t)desc, c, condition, a, b);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaWhere((WhereCudaDescriptor_t)desc, c, condition, a, b, stream);
        }
#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t)desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyWhereDescriptor((WhereCudaDescriptor_t)desc);
        }
#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}