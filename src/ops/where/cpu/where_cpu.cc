#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <numeric>

// 辅助函数声明
static void incrementOne(std::vector<int64_t>& indices,  uint64_t* shape, uint64_t ndim);
static uint64_t compactToFlat(const std::vector<int64_t>& indices, const int64_t* strides, uint64_t ndim);

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t condition,
                                          infiniopTensorDescriptor_t a,
                                          infiniopTensorDescriptor_t b,
                                          infiniopTensorDescriptor_t c) {
    if (!desc_ptr) return STATUS_MEMORY_NOT_ALLOCATED;
    if (condition->dt != U8 || a->dt != b->dt || a->dt != c->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!isValidBroadcastShape(condition, a, c) || !isValidBroadcastShape(b, a, c)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t data_size = std::accumulate(c->shape, c->shape + c->ndim, 1ULL, std::multiplies<uint64_t>());
    bool broadcasted = false;
    if (c->ndim != condition->ndim || c->ndim != a->ndim || c->ndim != b->ndim) {
        broadcasted = true;
    } else {
        for (uint64_t i = 0; i < c->ndim; ++i) {
            if (c->shape[i] != condition->shape[i] || c->shape[i] != a->shape[i] || c->shape[i] != b->shape[i]) {
                broadcasted = true;
                break;
            }
        }
    }

    WhereCpuDescriptor_t desc = new WhereCpuDescriptor;
    desc->device = DevCpu;
    desc->dtype = c->dt;
    desc->ndim = c->ndim;
    desc->data_size = data_size;
    desc->condition_strides = condition->strides;
    desc->a_strides = a->strides;
    desc->b_strides = b->strides;
    desc->c_shape = c->shape;
    desc->broadcasted = broadcasted;
    *desc_ptr = desc;

    return STATUS_SUCCESS;
}

template<typename T>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc,
                          void *c,
                          void const *condition,
                          void const *a,
                          void const *b) {
    auto condition_ = static_cast<const uint8_t *>(condition);
    auto a_ = static_cast<const T *>(a);
    auto b_ = static_cast<const T *>(b);
    auto c_ = static_cast<T *>(c);
    
    if (!desc->broadcasted) {
        for (uint64_t i = 0; i < desc->data_size; ++i) {
            c_[i] = condition_[i] ? a_[i] : b_[i];
        }
    } else {
        std::vector<int64_t> indices(desc->ndim);
        for (uint64_t i = 0; i < desc->data_size; ++i) {
            incrementOne(indices, desc->c_shape, desc->ndim);
            auto cond_idx = compactToFlat(indices, desc->condition_strides, desc->ndim);
            auto a_idx = compactToFlat(indices, desc->a_strides, desc->ndim);
            auto b_idx = compactToFlat(indices, desc->b_strides, desc->ndim);
            c_[i] = condition_[cond_idx] ? a_[a_idx] : b_[b_idx];
        }
    }
    return STATUS_SUCCESS;
}

// 辅助函数实现
static void incrementOne(std::vector<int64_t>& indices,  uint64_t* shape, uint64_t ndim) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        indices[i]++;
        if (indices[i] < shape[i]) break;
        indices[i] = 0;
    }
}

static uint64_t compactToFlat(const std::vector<int64_t>& indices, const int64_t* strides, uint64_t ndim) {
    uint64_t flat_index = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *c,
                          void const *condition,
                          void const *a,
                          void const *b) {
    if (desc->dtype == F32) {
        return where_cpu<float>(desc, c, condition, a, b);
    } else if (desc->dtype == F16) {
        return where_cpu<uint16_t>(desc, c, condition, a, b);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}