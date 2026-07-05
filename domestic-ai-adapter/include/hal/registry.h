#ifndef HAL_REGISTRY_H
#define HAL_REGISTRY_H

#include "device.h"
#include "executor.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>

namespace hal {

enum class BackendType {
    SHENGTEN,
    SIYUAN,
    CPU
};

class BackendFactory {
public:
    virtual ~BackendFactory() = default;
    virtual DevicePtr createDevice(int deviceId = 0) = 0;
    virtual ExecutorPtr createExecutor() = 0;
    virtual std::string getName() const = 0;
    virtual bool isAvailable() const = 0;
};

using FactoryPtr = std::shared_ptr<BackendFactory>;

class Registry {
public:
    static Registry& instance();
    void registerFactory(BackendType type, std::function<FactoryPtr()> creator);
    FactoryPtr createFactory(BackendType type);
    std::vector<BackendType> getAvailableBackends() const;

private:
    Registry() = default;
    std::unordered_map<BackendType, std::function<FactoryPtr()>> factories_;
};

} // namespace hal

#endif
