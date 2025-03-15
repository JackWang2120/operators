#ifndef WHERE_CUDA_H
#define WHERE_CUDA_H

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct WhereCudaDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t data_size;
    int64_t *condition_strides;
    int64_t  *a_strides;
    int64_t  *b_strides;
    uint64_t  *c_shape;
    bool broadcasted;
};

typedef struct WhereCudaDescriptor *WhereCudaDescriptor_t;

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t handle,
                                           WhereCudaDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t condition,
                                           infiniopTensorDescriptor_t a,
                                           infiniopTensorDescriptor_t b,
                                           infiniopTensorDescriptor_t c);

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
                           void *c,
                           void const *condition,
                           void const *a,
                           void const *b,
                           void *stream);

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc);

#endif // WHERE_CUDA_H