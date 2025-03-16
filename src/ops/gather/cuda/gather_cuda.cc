#include "gather_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                          GatherCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t input,
                                          infiniopTensorDescriptor_t indices,
                                          infiniopTensorDescriptor_t output,
                                          int64_t axis) {
    if (!desc_ptr) return STATUS_MEMORY_NOT_ALLOCATED;
    if (indices->dt != I64 || input->dt != output->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (axis < 0 || axis >= input->ndim) {
        return STATUS_BAD_PARAM;
    }

    uint64_t data_size = std::accumulate(output->shape, 
                                        output->shape + output->ndim,
                                        1ULL, 
                                        std::multiplies<uint64_t>());

    *desc_ptr = new GatherCudaDescriptor{
        DevNvGpu,
        input->dt,
        handle->device_id,
        input->ndim,
        data_size,
        axis,
        input->shape,
        input->strides,
        indices->shape,
        output->shape
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}