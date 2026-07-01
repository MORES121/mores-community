#ifndef HAL_TENSOR_H
#define HAL_TENSOR_H

#include <cstddef>
#include <vector>
#include <memory>

namespace hal {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64,
    UINT8,
};

class Tensor {
public:
    virtual ~Tensor() = default;
    virtual void* getData() = 0;
    virtual const void* getData() const = 0;
    virtual std::vector<size_t> getShape() const = 0;
    virtual DataType getDataType() const = 0;
    virtual size_t getSizeInBytes() const = 0;
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace hal

#endif
