cmake_minimum_required(VERSION 3.14)
project(domestic_ai_adapter VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hal STATIC src/hal/registry.cpp)
target_include_directories(hal PUBLIC include)

add_executable(test_hal src/test/test_hal.cpp)
target_link_libraries(test_hal hal)
