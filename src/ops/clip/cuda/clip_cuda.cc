#include "clip_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                          ClipCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t c,
                                          infiniopTensorDescriptor_t a,
                                          float* min_value,
                                          float* max_value) {
    if (c->dt != F16 && c->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (c->dt != a->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(a) || !is_contiguous(c)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t c_data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, handle->device_id);

    *desc_ptr = new ClipCudaDescriptor{
        DevNvGpu,
        c->dt,
        handle->device_id,
        c->ndim,
        c_data_size,
        static_cast<uint64_t>(prop.maxGridSize[0]),
        min_value,
        max_value,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}