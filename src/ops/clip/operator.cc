#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/clip_cuda.cuh"
#endif

 infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t b,
    infiniopTensorDescriptor_t a,
    float* min_value, float* max_value) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *)desc_ptr, b, a, min_value, max_value);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateClipDescriptor((CudaHandle_t) handle, (ClipCudaDescriptor_t *) desc_ptr, b, a, min_value, max_value);
        }
#endif  
        default:
            return STATUS_BAD_DEVICE;

    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc, void *b, void const *a, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuClip((ClipCpuDescriptor_t)desc, b, a, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaClip((ClipCudaDescriptor_t) desc, b, a, stream);
        }
#endif
        default:
            return STATUS_BAD_DEVICE;

    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyClipDescriptor((ClipCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyClipDescriptor((ClipCudaDescriptor_t) desc);
        }
#endif
        default:
            return STATUS_BAD_DEVICE;

    }
    return STATUS_BAD_DEVICE;
}