#ifndef __CUDA_CLIP_H__
#define __CUDA_CLIP_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>

struct ClipCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t max_grid_size;
    float* min_value;
    float* max_value;
};

typedef struct ClipCudaDescriptor *ClipCudaDescriptor_t;

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                          ClipCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t c,
                                          infiniopTensorDescriptor_t a,
                                          float* min_value,
                                          float* max_value);

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc);

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                          void *c, void const *a,
                          void *stream);

#endif // __CUDA_CLIP_H__