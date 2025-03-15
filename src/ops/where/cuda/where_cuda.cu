#include "where_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <numeric>

template<typename T>
__device__ uint64_t compactToFlat(
    const uint64_t* indices,
    const int64_t* strides,
    const uint64_t ndim) {
    uint64_t flat_index = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}


template<typename T>
__global__ void where_kernel(
    T* c,
    const bool* condition,
    const T* a,
    const T* b,
    const int64_t* condition_strides,
    const int64_t* a_strides,
    const int64_t* b_strides,
    uint64_t* c_shape,
    uint64_t ndim,
    uint64_t data_size,
    bool broadcasted) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= data_size) return;

    if (!broadcasted) {
        c[tid] = condition[tid] ? a[tid] : b[tid];
    } else {
        uint64_t indices[32];
        uint64_t temp = tid;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }
        const uint64_t cond_idx = compactToFlat<T>(indices, condition_strides, ndim);
        const uint64_t a_idx = compactToFlat<T>(indices, a_strides, ndim);
        const uint64_t b_idx = compactToFlat<T>(indices, b_strides, ndim);
        c[tid] = condition[cond_idx] ? a[a_idx] : b[b_idx];
    }
}


template<typename T>
void _where_nv_gpu(
    WhereCudaDescriptor_t desc,
    T* c,
    const bool* condition,
    const T* a,
    const T* b,
    void* stream) {
    if (desc->data_size == 0) return;

    const uint64_t block_size = 256;
    const uint64_t grid_size = (desc->data_size + block_size - 1) / block_size;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        where_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        c, condition, a, b,
        desc->condition_strides,
        desc->a_strides,
        desc->b_strides,
        desc->c_shape,
        desc->ndim,
        desc->data_size,
        desc->broadcasted
    );
}
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
            if (c->shape[i] != condition->shape[i] || c->shape[i] != a->shape[i] || c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }

    *desc_ptr = new WhereCudaDescriptor{
        DevNvGpu,
        c->dt,
        c->ndim,
        data_size,
        condition->strides,
        a->strides,
        b->strides,
        c->shape,
        broadcasted
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
                          void *c,
                          void const *condition,
                          void const *a,
                          void const *b,
                          void *stream) {
    if (!desc || !c || !condition || !a || !b) {
        return STATUS_BAD_PARAM;
    }
    if (desc->dtype == F32) {
        _where_nv_gpu<float>(desc,
                            static_cast<float *>(c),
                            static_cast<const bool *>(condition),
                            static_cast<const float *>(a),
                            static_cast<const float *>(b),
                            stream);
    } else if (desc->dtype == F16) {
        _where_nv_gpu<__half>(desc,
                             static_cast<__half *>(c),
                             static_cast<const bool *>(condition),
                             static_cast<const __half *>(a),
                             static_cast<const __half *>(b),
                             stream);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}