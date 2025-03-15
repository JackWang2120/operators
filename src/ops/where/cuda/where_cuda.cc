#include "where_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t handle,
                                         WhereCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t condition,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b,
                                         infiniopTensorDescriptor_t c) {
    if (!desc_ptr) return STATUS_MEMORY_NOT_ALLOCATED;
    if (condition->dt != U8 || a->dt != b->dt || a->dt != c->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!isValidBroadcastShape(condition, a, c) || !isValidBroadcastShape(b, a, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());
    bool broadcasted = false;
    if (c->ndim != condition->ndim || c->ndim != a->ndim || c->ndim != b->ndim) {
        broadcasted = true;
    } else {
        for (uint64_t i = 0; i < c->ndim; ++i) {
            if (c->shape[i] != condition->shape[i] || c->shape[i] != a->shape[i] || 
                c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }

    WhereCudaDescriptor_t desc = new WhereCudaDescriptor;
    desc->device = DevNvGpu;
    desc->dtype = c->dt;
    desc->ndim = c->ndim;
    desc->data_size = data_size;
    desc->condition_strides = condition->strides;
    desc->a_strides = a->strides;
    desc->b_strides = b->strides;
    desc->c_shape = c->shape;
    desc->broadcasted = broadcasted;
    *desc_ptr = desc;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}