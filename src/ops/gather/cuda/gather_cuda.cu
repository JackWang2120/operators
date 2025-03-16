#include "gather_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

template<typename T>
__device__ uint64_t compute_flat_index(const uint64_t* indices,
                                     const int64_t* strides,
                                     uint64_t ndim) {
    uint64_t flat_index = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}

template<typename T>
__global__ void gather_kernel(T* output,
                            const T* input,
                            const int64_t* indices,
                            const int64_t* input_shape,
                            const int64_t* input_strides,
                            const int64_t* indices_shape,
                            const int64_t* output_shape,
                            uint64_t ndim,
                            int64_t axis,
                            uint64_t data_size) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= data_size) return;

    uint64_t coords[32];  // 假设最大维度为32
    uint64_t temp = tid;

    // 计算输出坐标
    for (int64_t i = ndim - 1; i >= 0; --i) {
        coords[i] = temp % output_shape[i];
        temp /= output_shape[i];
    }

    // 使用indices替换axis维度的坐标
    coords[axis] = indices[coords[axis]];

    // 计算输入索引
    const uint64_t input_idx = compute_flat_index<T>(coords, input_strides, ndim);
    output[tid] = input[input_idx];
}

template<typename T>
void _gather_nv_gpu(GatherCudaDescriptor_t desc,
                    T* output,
                    const T* input,
                    const int64_t* indices,
                    void* stream) {
    if (desc->data_size == 0) return;

    const uint64_t block_size = 256;
    const uint64_t grid_size = (desc->data_size + block_size - 1) / block_size;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    gather_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        output,
        input,
        indices,
        desc->input_shape,
        desc->input_strides,
        desc->indices_shape,
        desc->output_shape,
        desc->ndim,
        desc->axis,
        desc->data_size
    );
}

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                           void *output,
                           void const *input,
                           void const *indices,
                           void *stream) {
    if (!desc || !output || !input || !indices) {
        return STATUS_BAD_PARAM;
    }

    checkCudaError(cudaSetDevice(desc->device_id));

    if (desc->dtype == F32) {
        _gather_nv_gpu<float>(desc,
                             static_cast<float*>(output),
                             static_cast<const float*>(input),
                             static_cast<const int64_t*>(indices),
                             stream);
    } else if (desc->dtype == F16) {
        _gather_nv_gpu<__half>(desc,
                              static_cast<__half*>(output),
                              static_cast<const __half*>(input),
                              static_cast<const int64_t*>(indices),
                              stream);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return STATUS_SUCCESS;
}