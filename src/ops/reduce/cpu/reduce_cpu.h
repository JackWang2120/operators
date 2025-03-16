#ifndef REDUCE_CPU_H
#define REDUCE_CPU_H

#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "operators.h"
#include <vector>

struct ReduceCpuDescriptor {
    Device device;
    DT dtype;
    bool use_1Dreduce;        // 是否使用1D reduce
    std::vector<int> axes;    // 要reduce的轴
    std::vector<int> out_of_axes; // 非reduce轴
    uint64_t ndim;            // 输入维度
    uint64_t axes_ndim;       // reduce轴数量
    uint64_t out_ndim;        // 输出维度
    uint64_t axes_size;       // reduce轴总大小
    uint64_t out_size;        // 输出总大小
    int64_t *strides;         // 输入步长
    std::vector<int> axes_strides;  // reduce轴步长
    std::vector<int> out_strides;   // 输出步长
    int reduce_op;            // 0:min, 1:max, 2:mean
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                         ReduceCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         int64_t *axes,
                                         uint64_t axes_ndim,
                                         int reduce_op);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                          void *y,
                          void const *x,
                          void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif // REDUCE_CPU_H