#ifndef GATHER_CUDA_H
#define GATHER_CUDA_H

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct GatherCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    int64_t axis;
    int64_t const *input_shape;
    int64_t const *input_strides;
    int64_t const *indices_shape;
    int64_t const *output_shape;
};

typedef struct GatherCudaDescriptor *GatherCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                          GatherCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t input,
                                          infiniopTensorDescriptor_t indices,
                                          infiniopTensorDescriptor_t output,
                                          int64_t axis);

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                           void *output,
                           void const *input,
                           void const *indices,
                           void *stream);

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc);

#endif // GATHER_CUDA_H