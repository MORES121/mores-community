#include "hal/tensor.h"
#include "hal/device.h"
#include "hal/executor.h"
#include "hal/registry.h"
#include <iostream>

using namespace hal;

int main() {
    std::cout << "=== HAL 接口编译测试 ===" << std::endl;
    std::cout << "tensor.h  : OK" << std::endl;
    std::cout << "device.h  : OK" << std::endl;
    std::cout << "executor.h: OK" << std::endl;
    std::cout << "registry.h: OK" << std::endl;
    std::cout << "\n所有头文件编译通过！" << std::endl;
    return 0;
}
