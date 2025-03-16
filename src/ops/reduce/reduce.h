#ifndef REDUCE_H
#define REDUCE_H

#include "export.h"
#include "operators.h"

// Reduce操作类型枚举
typedef enum {
    REDUCE_MIN = 0,
    REDUCE_MAX = 1,
    REDUCE_MEAN = 2
} ReduceMode;

// Reduce描述符结构
typedef struct ReduceDescriptor {
    Device device;
} ReduceDescriptor;

typedef ReduceDescriptor *infiniopReduceDescriptor_t;

// 创建Reduce描述符
__C  infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t* axes,
    int64_t axes_ndim,
    ReduceMode mode);

// 获取工作空间大小
__C  infiniopStatus_t infiniopGetReduceWorkspaceSize(
    infiniopReduceDescriptor_t desc,
    uint64_t *size);

// 执行Reduce操作
__C  infiniopStatus_t infiniopReduce(
    infiniopReduceDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *y,
    void const *x,
    void *stream);

// 销毁Reduce描述符
__C  infiniopStatus_t infiniopDestroyReduceDescriptor(
    infiniopReduceDescriptor_t desc);

#endif // REDUCE_H