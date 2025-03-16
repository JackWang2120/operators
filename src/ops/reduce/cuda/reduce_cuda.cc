#include "reduce_cuda.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                          ReduceCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          int *axes,
                                          uint64_t axes_ndim,
                                          int reduce_op) {
    // 检查输入tensor
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    // 检查数据类型
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // 创建cuDNN描述符
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnReduceTensorDescriptor_t reduce_desc;
    
    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnnError(cudnnCreateReduceTensorDescriptor(&reduce_desc));

    // 设置tensor描述符
    cudnnDataType_t compute_type;
    if (x->dt == F16) {
        compute_type = CUDNN_DATA_HALF;
    } else {
        compute_type = CUDNN_DATA_FLOAT;
    }

    // 配置输入描述符
    if (x->ndim <= 4) {
        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < x->ndim; i++) {
            dims[4 - x->ndim + i] = x->shape[i];
        }
        checkCudnnError(cudnnSetTensor4dDescriptor(x_desc,
                                                  CUDNN_TENSOR_NCHW,
                                                  compute_type,
                                                  dims[0], dims[1],
                                                  dims[2], dims[3]));
    } else {
        checkCudnnError(cudnnSetTensorNdDescriptor(x_desc,
                                                  compute_type,
                                                  x->ndim,
                                                  x->shape,
                                                  x->strides));
    }

    // 配置输出描述符
    if (y->ndim <= 4) {
        int dims[4] = {1, 1, 1, 1};
        for (int i = 0; i < y->ndim; i++) {
            dims[4 - y->ndim + i] = y->shape[i];
        }
        checkCudnnError(cudnnSetTensor4dDescriptor(y_desc,
                                                  CUDNN_TENSOR_NCHW,
                                                  compute_type,
                                                  dims[0], dims[1],
                                                  dims[2], dims[3]));
    } else {
        checkCudnnError(cudnnSetTensorNdDescriptor(y_desc,
                                                  compute_type,
                                                  y->ndim,
                                                  y->shape,
                                                  y->strides));
    }

    // 配置reduce描述符
    cudnnReduceTensorOp_t reduce_op_t;
    switch(reduce_op) {
        case REDUCE_MIN: reduce_op_t = CUDNN_REDUCE_TENSOR_MIN; break;
        case REDUCE_MAX: reduce_op_t = CUDNN_REDUCE_TENSOR_MAX; break;
        case REDUCE_MEAN: reduce_op_t = CUDNN_REDUCE_TENSOR_AVG; break;
        default: return STATUS_BAD_PARAM;
    }

    checkCudnnError(cudnnSetReduceTensorDescriptor(reduce_desc,
                                                  reduce_op_t,
                                                  compute_type,
                                                  CUDNN_NOT_PROPAGATE_NAN,
                                                  CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                  CUDNN_32BIT_INDICES));

    // 获取工作空间大小
    size_t workspace_size;
    checkCudnnError(cudnnGetReductionWorkspaceSize(handle->cudnn_handle,
                                                  reduce_desc,
                                                  x_desc,
                                                  y_desc,
                                                  &workspace_size));

    // 创建描述符
    *desc_ptr = new ReduceCudaDescriptor{
        DevNvGpu,
        x->dt,
        handle->device_id,
        handle->cudnn_handles,
        x_desc,
        y_desc,
        reduce_desc,
        workspace_size
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetReduceWorkspaceSize(ReduceCudaDescriptor_t desc,
                                          uint64_t *size) {
    if (!desc || !size) {
        return STATUS_BAD_PARAM;
    }
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc) {
    if (!desc) {
        return STATUS_BAD_PARAM;
    }
    
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyReduceTensorDescriptor(desc->reduce_desc));
    
    delete desc;
    return STATUS_SUCCESS;
}