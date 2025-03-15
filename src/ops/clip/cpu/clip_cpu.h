#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t b_data_size;
    float min_value;
    float max_value;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

 infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                        ClipCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t b,
                                        infiniopTensorDescriptor_t a,
                                        float* min_value, float* max_value);

 infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                        void *b, void const *a, 
                        void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
