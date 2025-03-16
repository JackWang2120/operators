#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce/reduce_min.h"

struct _ReduceMinDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
    uint64_t workspace_size;
};

typedef struct _ReduceMinDescriptor *_ReduceMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMinDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t *axes,
    uint64_t axes_ndim) {
    
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc,
                                              y, x, axes, axes_ndim,
                                              REDUCE_MIN), STATUS_SUCCESS);

    uint64_t workspace_size = 0;
    CHECK_STATUS(infiniopGetReduceWorkspaceSize(reduce_desc, &workspace_size),
                STATUS_SUCCESS);

    *(_ReduceMinDescriptor_t *)desc_ptr = new _ReduceMinDescriptor{
        handle->device,
        reduce_desc,
        workspace_size
    };

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetReduceMinWorkspaceSize(
    infiniopReduceMinDescriptor_t desc,
    uint64_t *size) {
    
    *size = ((_ReduceMinDescriptor_t)desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMin(
    infiniopReduceMinDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *y,
    void const *x,
    void *stream) {
    
    auto _desc = (_ReduceMinDescriptor_t)desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    CHECK_STATUS(infiniopReduce(_desc->reduce_desc,
                              workspace, workspace_size,
                              y, x, stream),
                STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(
    infiniopReduceMinDescriptor_t desc) {
    
    CHECK_STATUS(infiniopDestroyReduceDescriptor(
        ((_ReduceMinDescriptor_t)desc)->reduce_desc),
        STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}