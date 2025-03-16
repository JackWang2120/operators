#include "reduce_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <numeric>
#include <algorithm>

template<typename Tdata>
infiniopStatus_t reduce_cpu_1D(ReduceCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto data_size_ = desc->out_size * desc->axes_size;

    if constexpr (std::is_same<Tdata, uint16_t>::value) {
        switch (desc->reduce_op) {
            case 0: { // ReduceMin
                float result = std::numeric_limits<float>::max();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::min(result, f16_to_f32(x_[i]));
                }
                y_[0] = f32_to_f16(result);
                break;
            }
            case 1: { // ReduceMax
                float result = std::numeric_limits<float>::lowest();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::max(result, f16_to_f32(x_[i]));
                }
                y_[0] = f32_to_f16(result);
                break;
            }
            case 2: { // ReduceMean
                float sum = 0;
                for (uint64_t i = 0; i < data_size_; i++) {
                    sum += f16_to_f32(x_[i]);
                }
                y_[0] = f32_to_f16(sum / data_size_);
                break;
            }
        }
    } else {
        switch (desc->reduce_op) {
            case 0: { // ReduceMin
                Tdata result = std::numeric_limits<Tdata>::max();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::min(result, x_[i]);
                }
                y_[0] = result;
                break;
            }
            case 1: { // ReduceMax
                Tdata result = std::numeric_limits<Tdata>::lowest();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::max(result, x_[i]);
                }
                y_[0] = result;
                break;
            }
            case 2: { // ReduceMean
                Tdata sum = 0;
                for (uint64_t i = 0; i < data_size_; i++) {
                    sum += x_[i];
                }
                y_[0] = sum / data_size_;
                break;
            }
        }
    }
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    
    #pragma omp parallel for
    for (uint64_t i = 0; i < desc->out_size; ++i) {
        uint64_t idx = 0;
        uint64_t temp_i = i;
        
        // 计算输出索引
        for (uint64_t j = 0; j < desc->out_ndim; ++j) {
            idx += temp_i / desc->out_strides[j] * desc->strides[desc->out_of_axes[j]];
            temp_i %= desc->out_strides[j];
        }

        float result;
        switch (desc->reduce_op) {
            case 0: result = std::numeric_limits<float>::max(); break;
            case 1: result = std::numeric_limits<float>::lowest(); break;
            case 2: result = 0; break;
        }

        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            for (uint64_t j = 0; j < desc->axes_size; ++j) {
                uint64_t idx_ = idx;
                uint64_t temp_j = j;
                for (uint64_t k = 0; k < desc->axes_ndim; ++k) {
                    idx_ += temp_j / desc->axes_strides[k] * desc->strides[desc->axes[k]];
                    temp_j %= desc->axes_strides[k];
                }

                float val = f16_to_f32(x_[idx_]);
                switch (desc->reduce_op) {
                    case 0: result = std::min(result, val); break;
                    case 1: result = std::max(result, val); break;
                    case 2: result += val; break;
                }
            }
            
            if (desc->reduce_op == 2) { // Mean
                result /= desc->axes_size;
            }
            y_[i] = f32_to_f16(result);
        } else {
            // FP32处理
            for (uint64_t j = 0; j < desc->axes_size; ++j) {
                uint64_t idx_ = idx;
                uint64_t temp_j = j;
                for (uint64_t k = 0; k < desc->axes_ndim; ++k) {
                    idx_ += temp_j / desc->axes_strides[k] * desc->strides[desc->axes[k]];
                    temp_j %= desc->axes_strides[k];
                }

                switch (desc->reduce_op) {
                    case 0: result = std::min(result, x_[idx_]); break;
                    case 1: result = std::max(result, x_[idx_]); break;
                    case 2: result += x_[idx_]; break;
                }
            }
            
            if (desc->reduce_op == 2) { // Mean
                result /= desc->axes_size;
            }
            y_[i] = result;
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->dtype == F16) {
        if (desc->use_1Dreduce) {
            return reduce_cpu_1D<uint16_t>(desc, y, x);
        } else {
            return reduce_cpu<uint16_t>(desc, y, x);
        }   
    }
    if (desc->dtype == F32) {
        if (desc->use_1Dreduce) {
            return reduce_cpu_1D<float>(desc, y, x);
        } else {
            return reduce_cpu<float>(desc, y, x);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                         ReduceCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         int *axes,
                                         uint64_t axes_ndim,
                                         int reduce_op) {
    // 参数检查
    if (!desc_ptr || !y || !x || !axes) {
        return STATUS_BAD_PARAM;
    }

    // 数据类型检查
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // 创建描述符 
    auto desc = new ReduceCpuDescriptor();
    desc->device = DevCpu;
    desc->dtype = x->dt;
    desc->reduce_op = reduce_op;
    desc->ndim = x->ndim;
    desc->axes_ndim = axes_ndim;
    desc->axes.assign(axes, axes + axes_ndim);
    
    // 计算输入/输出形状
    uint64_t out_ndim = desc->ndim - axes_ndim;
    desc->out_ndim = out_ndim;
    
    // 计算步长
    desc->strides = new int64_t[desc->ndim];
    std::memcpy(desc->strides, x->strides, desc->ndim * sizeof(int64_t));

    *desc_ptr = desc;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    if (!desc) return STATUS_BAD_PARAM;
    delete[] desc->strides;
    delete desc;
    return STATUS_SUCCESS;
}