#ifndef GATHER_CPU_H
#define GATHER_CPU_H

#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "operators.h"

typedef struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t data_size;
    int64_t axis;
    uint64_t  *input_shape;
    int64_t  *input_strides;
    uint64_t  *indices_shape; 
    uint64_t  *output_shape;
} GatherCpuDescriptor;

typedef GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
                                         GatherCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t indices,
                                         infiniopTensorDescriptor_t output,
                                         int64_t axis);

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                          void *output,
                          void const *input,
                          void const *indices);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif // GATHER_CPU_H