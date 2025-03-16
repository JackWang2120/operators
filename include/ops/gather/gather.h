#ifndef GATHER_H
#define GATHER_H

#include "../../export.h"
#include "../../operators.h"

typedef struct GatherDescriptor {
    Device device;
} GatherDescriptor;

typedef GatherDescriptor *infiniopGatherDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t indices,
    infiniopTensorDescriptor_t output,
    int64_t axis);

__C __export infiniopStatus_t infiniopGather(
    infiniopGatherDescriptor_t desc,
    void *output,
    void const *input,
    void const *indices,
    void *stream);

__C __export infiniopStatus_t infiniopDestroyGatherDescriptor(
    infiniopGatherDescriptor_t desc);

#endif // GATHER_H