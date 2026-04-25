#include "hal/registry.h"
#include <iostream>
#include <vector>
#include <functional>

namespace hal {

Registry& Registry::instance() {
    static Registry registry;
    return registry;
}

void Registry::registerFactory(BackendType type, std::function<FactoryPtr()> creator) {
    factories_[type] = creator;
    std::cout << "[Registry] 注册后端: " << static_cast<int>(type) << std::endl;
}

FactoryPtr Registry::createFactory(BackendType type) {
    auto it = factories_.find(type);
    if (it != factories_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<BackendType> Registry::getAvailableBackends() const {
    std::vector<BackendType> types;
    for (const auto& pair : factories_) {
        types.push_back(pair.first);
    }
    return types;
}

} // namespace hal
