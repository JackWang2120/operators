#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "clip_cuda.cuh"

template<typename T>
__global__ void clip_kernel(
    T *c,
    const T *a,
    uint64_t data_size,
    float min_value,
    float max_value) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < data_size) {
       // c[idx] = max(min(static_cast<float>(a[idx]), max_value), min_value);
     
        if constexpr (std::is_same<T, __half>::value) {
            c[idx] = __float2half(fmaxf(fminf(__half2float(a[idx]), max_value), min_value));
        } else {
            c[idx] = static_cast<float>(fmaxf(fminf(static_cast<float>(a[idx]), max_value), min_value));
        }
    }
}

template<typename Tdata>
void _clip_nv_gpu(ClipCudaDescriptor_t desc, Tdata *c, const Tdata *a, uint64_t data_size, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t batch = gridDims.x*blockDims.x;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#pragma unroll 
    for(uint64_t i = 0; i < data_size; i += batch) {
        clip_kernel<<<gridDims, blockDims, 0, cuda_stream>>>(c + i, a + i, std::min(data_size - i, batch), *desc->min_value, *desc->max_value);
    }
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                          void *c, void const *a,
                          void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        _clip_nv_gpu<__half>(desc, static_cast<__half *>(c), static_cast<const __half *>(a), desc->c_data_size, stream);
    } else if (desc->dtype == F32) {
        _clip_nv_gpu<float>(desc, static_cast<float *>(c), static_cast<const float *>(a), desc->c_data_size, stream);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    return STATUS_SUCCESS;
}