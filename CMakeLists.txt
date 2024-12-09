cmake_minimum_required(VERSION 3.20)
project(YourApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES}) 