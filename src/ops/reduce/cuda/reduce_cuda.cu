#include "reduce_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t reduce_nv_gpu(ReduceCudaDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *y,
                              void const *x,
                              void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    
    float alpha = 1.0f;
    float beta = 0.0f;

    checkCudnnError(use_cudnn(desc->cudnn_handles,
                             desc->device_id,
                             static_cast<cudaStream_t>(stream),
                             [&](cudnnHandle_t handle) {
        return cudnnReduceTensor(handle,
                               desc->reduce_desc,
                               nullptr,
                               0,
                               workspace,
                               workspace_size,
                               &alpha,
                               desc->x_desc,
                               x,
                               &beta,
                               desc->y_desc,
                               y);
    }));

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *y,
                           void const *x,
                           void *stream) {
    if (!desc || !y || !x) {
        return STATUS_BAD_PARAM;
    }

    if (desc->dtype != F16 && desc->dtype != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    return reduce_nv_gpu(desc, workspace, workspace_size, y, x, stream);
}