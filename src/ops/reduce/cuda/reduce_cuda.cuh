#ifndef REDUCE_CUDA_H
#define REDUCE_CUDA_H

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cudnn.h>

struct ReduceCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles;
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnReduceTensorDescriptor_t reduce_desc;
    uint64_t workspace_size;
};

typedef ReduceCudaDescriptor *ReduceCudaDescriptor_t;

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                          ReduceCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          int *axes,
                                          uint64_t axes_ndim,
                                          int reduce_op);

infiniopStatus_t cudaGetReduceWorkspaceSize(ReduceCudaDescriptor_t desc, 
                                          uint64_t *size);

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
                           void *workspace,
                           uint64_t workspace_size,
                           void *y,
                           void const *x,
                           void *stream);

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc);

#endif