#ifndef HAL_DEVICE_H
#define HAL_DEVICE_H

#include <string>
#include <vector>
#include <memory>

namespace hal {

enum class DeviceType {
    SHENGTEN,
    SIYUAN,
    CPU
};

struct DeviceInfo {
    std::string name;
    std::string version;
    DeviceType type;
    size_t memoryTotal;
    size_t memoryAvailable;
    int computeUnits;
};

class Device {
public:
    virtual ~Device() = default;
    virtual bool init() = 0;
    virtual void shutdown() = 0;
    virtual DeviceInfo getInfo() const = 0;
    virtual int getDeviceId() const = 0;
    virtual void synchronize() = 0;
    virtual bool isInitialized() const = 0;
};

using DevicePtr = std::shared_ptr<Device>;

} // namespace hal

#endif
