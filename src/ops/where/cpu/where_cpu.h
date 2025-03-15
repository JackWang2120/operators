#ifndef WHERE_CPU_H
#define WHERE_CPU_H

#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "operators.h"

typedef struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t data_size;
    int64_t  *condition_strides;
    int64_t  *a_strides; 
    int64_t  *b_strides;
    uint64_t  *c_shape;
    bool broadcasted;
} WhereCpuDescriptor;

typedef WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t condition,
                                          infiniopTensorDescriptor_t a,
                                          infiniopTensorDescriptor_t b,
                                          infiniopTensorDescriptor_t c);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *c,
                          void const *condition,
                          void const *a,
                          void const *b);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);

#endif // WHERE_CPU_H