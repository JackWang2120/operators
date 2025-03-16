#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <numeric>

static uint64_t compute_flat_index(int64_t* indices,
                                 const int64_t* strides,
                                 uint64_t ndim) {
    uint64_t flat_index = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}

template<typename T>
void gather_cpu(GatherCpuDescriptor_t desc,
                T* output,
                const T* input,
                const int64_t* indices) {
    // 用于存储当前坐标
    std::vector<int64_t> coords(desc->ndim);
    
    // 遍历所有输出元素
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        uint64_t temp = i;
        
        // 计算输出坐标
        for (int64_t j = desc->ndim - 1; j >= 0; --j) {
            coords[j] = temp % desc->output_shape[j];
            temp /= desc->output_shape[j];
        }
        
        // 使用indices替换axis维度的坐标
        coords[desc->axis] = indices[coords[desc->axis]];
        
        // 计算输入索引并复制数据
        uint64_t input_idx = compute_flat_index(coords.data(), 
                                              desc->input_strides, 
                                              desc->ndim);
        output[i] = input[input_idx];
    }
}

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
                                         GatherCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t indices,
                                         infiniopTensorDescriptor_t output,
                                         int64_t axis) {
    if (!desc_ptr || !input || !indices || !output) {
        return STATUS_BAD_PARAM;
    }
    if (indices->dt != I64 || input->dt != output->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (axis < 0 || axis >= input->ndim) {
        return STATUS_BAD_PARAM;
    }

    uint64_t data_size = std::accumulate(output->shape, 
                                        output->shape + output->ndim,
                                        1ULL, 
                                        std::multiplies<uint64_t>());

    GatherCpuDescriptor_t desc = new GatherCpuDescriptor;
    desc->device = DevCpu;
    desc->dtype = input->dt;
    desc->ndim = input->ndim;
    desc->data_size = data_size;
    desc->axis = axis;
    desc->input_shape = input->shape;
    desc->input_strides = input->strides;
    desc->indices_shape = indices->shape;
    desc->output_shape = output->shape;
    *desc_ptr = desc;

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                          void *output,
                          void const *input,
                          void const *indices) {
    if (desc->dtype == F32) {
        gather_cpu<float>(desc,
                         static_cast<float*>(output),
                         static_cast<const float*>(input),
                         static_cast<const int64_t*>(indices));
    } else if (desc->dtype == F16) {
        gather_cpu<uint16_t>(desc,
                            static_cast<uint16_t*>(output),
                            static_cast<const uint16_t*>(input),
                            static_cast<const int64_t*>(indices));
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}