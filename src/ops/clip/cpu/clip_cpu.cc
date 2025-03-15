#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <iostream>


 infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t b,
                                         infiniopTensorDescriptor_t a,
                                         float* min_value, float* max_value) {
    if (!is_contiguous(b) || !is_contiguous(a)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (b->dt != F16 && b->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (b->dt != a->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t b_data_size = std::accumulate(b->shape, b->shape + b->ndim, 1ULL, std::multiplies<uint64_t>());
    float minVal = min_value?*min_value:std::numeric_limits<float>::lowest();
    float maxVal = max_value?*max_value:std::numeric_limits<float>::max();
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        b->dt,
        b_data_size,
        minVal,
        maxVal,
    };
    return STATUS_SUCCESS;
}

template <typename T>
infiniopStatus_t cpu_Clip(ClipCpuDescriptor_t desc,
                             void *b, void const *a) {
    auto b_data = reinterpret_cast<T *>(b);
    auto a_data = reinterpret_cast<T const *>(a);
    float min_value = desc->min_value;
    float max_value = desc->max_value;
    // std::cout<<"min_value: "<<f16_to_f32(f32_to_f16(min_value))<<std::endl;
    // std::cout<<"max_value: "<<f16_to_f32(f32_to_f16(max_value))<<std::endl;
#pragma omp parallel    
    for (uint64_t i = 0; i < desc->b_data_size; ++i) {
        if constexpr (std::is_same<T,uint16_t>::value) {
            b_data[i] = f32_to_f16(std::min(max_value, std::max(min_value, f16_to_f32(a_data[i]))));
            // auto f32_val = f16_to_f32(a_data[i]);
            // //std::cout<<"f32_val: "<<f32_val<<std::endl;
            // f32_val = f32_val<min_value?min_value:f32_val;
            // f32_val = f32_val>max_value?max_value:f32_val;
            // b_data[i] = f32_to_f16(f32_val);
            // std::cout<<"b_data[i]: "<<b_data[i]<<std::endl;
            std::cout<<"min_value: "<<min_value<<std::endl;
            std::cout<<"max_value: "<<max_value<<std::endl;

        } else {
            b_data[i] = std::min(max_value, std::max(min_value, a_data[i]));
        }
    }
    return STATUS_SUCCESS;
}
 infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *b, void const *a, 
                         void *stream) {

    if (desc->dtype == F32) {
        return cpu_Clip<float>(desc, b, a);
    } else if (desc->dtype == F16) {
        return cpu_Clip<uint16_t>(desc, b, a);
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    return STATUS_SUCCESS;
}

 infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    if (!desc) return STATUS_MEMORY_NOT_ALLOCATED;
    delete desc;
    return STATUS_SUCCESS;
}