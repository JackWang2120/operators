### Clip 算子

#### 基本属性
| 描述 | 对输入张量的每个元素进行裁剪，使其值在指定的最小值和最大值之间 |
| ---- | ------------------------------------------------------------ |
| 是否支持原地（in-place）计算 | 是 |
| 是否需要额外工作空间（workspace） | 否 |

公式为：

$$C = \max(\min(A, \text{max_value}), \text{min_value})$$

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle, infiniopClipDescriptor_t *desc_ptr, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t c_desc, float min_value, float max_value);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| c_desc | 输出张量 C 的描述。形状与 A 相同，类型与 A 相同 |
| min_value | 裁剪的最小值 |
| max_value | 裁剪的最大值 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc, void* c, void const* a, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);
```

### Where 算子

#### 基本属性
| 描述 | 根据条件张量选择输入张量 A 或 B 的元素 |
| ---- | -------------------------------------- |
| 是否支持原地（in-place）计算 | 否 |
| 是否需要额外工作空间（workspace） | 否 |

公式为：

$$C = \begin{cases} A, & \text{if } \text{condition} \\ B, & \text{otherwise} \end{cases}$$

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle, infiniopWhereDescriptor_t *desc_ptr, infiniopTensorDescriptor_t condition_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| condition_desc | 条件张量的描述。形状与 A、B 相同，类型为 bool |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| b_desc | 输入张量 B 的描述。形状与 A 相同，类型与 A 相同 |
| c_desc | 输出张量 C 的描述。形状与 A 相同，类型与 A 相同 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc, void* c, void const* condition, void const* a, void const* b, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);
```

### Gather 算子

#### 基本属性
| 描述 | 根据索引张量从输入张量中收集元素 |
| ---- | -------------------------------- |
| 是否支持原地（in-place）计算 | 否 |
| 是否需要额外工作空间（workspace） | 否 |

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle, infiniopGatherDescriptor_t *desc_ptr, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t indices_desc, infiniopTensorDescriptor_t c_desc, int axis);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| indices_desc | 索引张量的描述。形状可以为任意形状，类型为 int32 |
| c_desc | 输出张量 C 的描述。形状根据索引张量和轴确定，类型与 A 相同 |
| axis | 收集元素的轴 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc, void* c, void const* a, void const* indices, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc);
```

### ReduceMin 算子

#### 基本属性
| 描述 | 沿着指定的轴计算输入张量的最小值 |
| ---- | -------------------------------- |
| 是否支持原地（in-place）计算 | 否 |
| 是否需要额外工作空间（workspace） | 否 |

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateReduceMinDescriptor(infiniopHandle_t handle, infiniopReduceMinDescriptor_t *desc_ptr, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t c_desc, int axis);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| c_desc | 输出张量 C 的描述。形状根据轴确定，类型与 A 相同 |
| axis | 计算最小值的轴 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopReduceMin(infiniopReduceMinDescriptor_t desc, void* c, void const* a, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyReduceMinDescriptor(infiniopReduceMinDescriptor_t desc);
```

### ReduceMax 算子

#### 基本属性
| 描述 | 沿着指定的轴计算输入张量的最大值 |
| ---- | -------------------------------- |
| 是否支持原地（in-place）计算 | 否 |
| 是否需要额外工作空间（workspace） | 否 |

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle, infiniopReduceMaxDescriptor_t *desc_ptr, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t c_desc, int axis);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| c_desc | 输出张量 C 的描述。形状根据轴确定，类型与 A 相同 |
| axis | 计算最大值的轴 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc, void* c, void const* a, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);
```

### ReduceMean 算子

#### 基本属性
| 描述 | 沿着指定的轴计算输入张量的平均值 |
| ---- | -------------------------------- |
| 是否支持原地（in-place）计算 | 否 |
| 是否需要额外工作空间（workspace） | 否 |

#### 接口定义

##### 创建算子描述
```C
infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle, infiniopReduceMeanDescriptor_t *desc_ptr, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t c_desc, int axis);
```

**参数说明**
| 参数 | 说明 |
| ---- | ---- |
| handle | 硬件控柄 |
| desc_ptr | 算子描述符的地址 |
| a_desc | 输入张量 A 的描述。形状可以为任意形状，类型可以为 fp16 或 fp32 |
| c_desc | 输出张量 C 的描述。形状根据轴确定，类型与 A 相同 |
| axis | 计算平均值的轴 |

**返回值**
| 返回值 | 说明 |
| ---- | ---- |
| STATUS_SUCCESS | 成功 |
| STATUS_BAD_PARAM | 参数张量不统一 |
| STATUS_BAD_TENSOR_DTYPE | 输入输出张量类型不被支持 |
| STATUS_BAD_TENSOR_SHAPE | 张量形状不符合要求 |
| STATUS_MEMORY_NOT_ALLOCATED | 描述符地址不合法 |

##### 计算
```C
infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc, void* c, void const* a, void* stream);
```

##### 删除算子描述
```C
infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc);
```
