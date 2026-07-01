#ifndef HAL_EXECUTOR_H
#define HAL_EXECUTOR_H

#include "tensor.h"
#include "device.h"
#include <string>
#include <vector>
#include <memory>

namespace hal {

class Executor {
public:
    virtual ~Executor() = default;
    virtual bool loadModel(const std::string& modelPath) = 0;
    virtual bool infer(const TensorPtr& input, TensorPtr& output) = 0;
    virtual bool inferBatch(const std::vector<TensorPtr>& inputs,
                            std::vector<TensorPtr>& outputs) = 0;
    virtual DevicePtr getDevice() const = 0;
    virtual std::string getModelName() const = 0;
    virtual std::vector<size_t> getInputShape() const = 0;
    virtual std::vector<size_t> getOutputShape() const = 0;
};

using ExecutorPtr = std::shared_ptr<Executor>;

} // namespace hal

#endif
